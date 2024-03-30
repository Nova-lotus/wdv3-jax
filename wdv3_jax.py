import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import flax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image
from simple_parsing import field, parse_known_args

import Models

MODEL_REPO_MAP = {
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
}


@flax.struct.dataclass
class PredModel:
    apply_fun: Callable = flax.struct.field(pytree_node=False)
    params: Any = flax.struct.field(pytree_node=True)

    def jit_predict(self, x):
        # Not actually JITed since this is a single shot script,
        # but this is the function you would decorate with @jax.jit
        x = x / 127.5 - 1
        x = self.apply_fun(self.params, x, train=False)
        x = flax.linen.sigmoid(x)
        x = jax.numpy.float32(x)
        return x

    def predict(self, x):
        preds = self.jit_predict(x)
        preds = jax.device_get(preds)
        preds = preds[0]
        return preds


def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    # convert to RGB/RGBA if not already (deals with palette images etc.)
    if image.mode not in ["RGB", "RGBA"]:
        image = (
            image.convert("RGBA")
            if "transparency" in image.info
            else image.convert("RGB")
        )
    # convert RGBA to RGB with white background
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    # get the largest dimension so we can pad to a square
    px = max(image.size)
    # pad to square with white background
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas


def pil_resize(image: Image.Image, target_size: int) -> Image.Image:
    # Resize
    max_dim = max(image.size)
    if max_dim != target_size:
        image = image.resize(
            (target_size, target_size),
            Image.BICUBIC,
        )
    return image


@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]


def load_labels_hf(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> LabelData:
    try:
        csv_path = hf_hub_download(
            repo_id=repo_id,
            filename="selected_tags.csv",
            revision=revision,
            token=token,
        )
        csv_path = Path(csv_path).resolve()
    except HfHubHTTPError as e:
        raise FileNotFoundError(
            f"selected_tags.csv failed to download from {repo_id}"
        ) from e

    df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
    tag_data = LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )

    return tag_data


def load_model_hf(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> PredModel:
    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.msgpack",
        revision=revision,
        token=token,
    )

    model_config = hf_hub_download(
        repo_id=repo_id,
        filename="sw_jax_cv_config.json",
        revision=revision,
        token=token,
    )

    with open(weights_path, "rb") as f:
        data = f.read()

    restored = flax.serialization.msgpack_restore(data)["model"]
    variables = {"params": restored["params"], **restored["constants"]}

    with open(model_config) as f:
        model_config = json.loads(f.read())

    model_name = model_config["model_name"]
    model_builder = Models.model_registry[model_name]()
    model = model_builder.build(
        config=model_builder,
        **model_config["model_args"],
    )
    model = PredModel(model.apply, params=variables)
    return model, model_config["image_size"]


def get_tags(
    probs: Any,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
):
    # Ensure probs is a list or array
    probs = [probs]

    # Check that the length of probs matches the expected length
    if len(probs[0]) != len(labels.names):
        raise ValueError("Length of probs does not match length of labels.names")

    # Convert indices+probs to labels
    probs = list(zip(labels.names, probs[0]))

    # First 4 labels are actually ratings
    rating_labels = dict([probs[i] for i in labels.rating])

    # General labels, pick any where prediction confidence > threshold
    gen_labels = [probs[i] for i in labels.general]
    gen_labels = dict([x for x in gen_labels if x[1] > gen_threshold])
    gen_labels = dict(
        sorted(
            gen_labels.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    # Character labels, pick any where prediction confidence > threshold
    char_labels = [probs[i] for i in labels.character]
    char_labels = dict([x for x in char_labels if x[1] > char_threshold])
    char_labels = dict(
        sorted(
            char_labels.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    # Combine general and character labels, sort by confidence
    combined_names = [x for x in gen_labels]
    combined_names.extend([x for x in char_labels])

    # Convert to a string suitable for use as a training caption
    caption = ", ".join(combined_names)
    taglist = caption.replace("_", " ").replace("(", "\(").replace(")", "\)")

    return caption, taglist, rating_labels, char_labels, gen_labels


@dataclass
class ScriptOptions:
    image_file: list[str] = field(positional=True, default_factory=list)
    model: str = field(default="vit")
    gen_threshold: float = field(default=0.35)
    char_threshold: float = field(default=0.75)


def main(opts: ScriptOptions, batch_size: int = 32):
    repo_id = MODEL_REPO_MAP.get(opts.model)
    image_paths = [Path(p).resolve() for p in opts.image_file]
    if not all(p.is_file() for p in image_paths):
        raise FileNotFoundError(f"One or more image files not found.")

    print(f"Loading model '{opts.model}' from '{repo_id}'...")
    model, target_size = load_model_hf(repo_id=repo_id)

    print("Loading tag list...")
    labels: LabelData = load_labels_hf(repo_id=repo_id)

    print("Loading images and preprocessing...")
    inputs = []
    for image_path in image_paths:
        img_input: Image.Image = Image.open(image_path)
        img_input = pil_ensure_rgb(img_input)
        img_input = pil_pad_square(img_input)
        img_input = pil_resize(img_input, target_size)
        img_input = np.array(img_input)
        img_input = img_input[..., ::-1]  # NHWC image RGB to BGR
        inputs.append(img_input)

    # Enable GPU acceleration in JAX
    jax.config.update('jax_platform_name', 'gpu')

    print("Running inference...")
    batched_inputs = np.array_split(inputs, len(inputs) // batch_size + 1)
    all_outputs = []
    for batch in batched_inputs:
        batch_inputs = jnp.array(batch)
        outputs = model.predict(batch_inputs)
        all_outputs.extend(outputs)

    print("Processing results...")
    for i, output in enumerate(all_outputs):
        image_path = image_paths[i]
        caption, taglist, ratings, character, general = get_tags(
            probs=output,
            labels=labels,
            gen_threshold=opts.gen_threshold,
            char_threshold=opts.char_threshold,
        )

        print(f"Results for {image_path.name}:")
        print("--------")
        print(f"Caption: {caption}")
        print("--------")
        print(f"Tags: {taglist}")
        # Print other results if needed

    print("Done!")


if __name__ == "__main__":
    opts, _ = parse_known_args(ScriptOptions)
    if opts.model not in MODEL_REPO_MAP:
        print(f"Available models: {list(MODEL_REPO_MAP.keys())}")
        raise ValueError(f"Unknown model name '{opts.model}'")
    main(opts)
