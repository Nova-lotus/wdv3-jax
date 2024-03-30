import os
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count

IMAGE_FOLDER = r''  # Set the path to the folder containing the images

def process_batch(batch_paths, model='swinv2', gen_threshold=0.1, char_threshold=0.1): #Set the path to wdv3_jax.py
    command = ['python', r'wdv3_jax.py', '--model', model, '--gen_threshold', str(gen_threshold), '--char_threshold', str(char_threshold)] + batch_paths
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout

def save_tags(filename, tags):
    text_filename = os.path.splitext(filename)[0] + '.txt'
    with open(text_filename, 'w', encoding='utf-8') as f:
        f.write(tags)

def process_batch_wrapper(args):
    batch_paths, model, gen_threshold, char_threshold = args
    output = process_batch(batch_paths, model, gen_threshold, char_threshold)
    output_lines = output.strip().split('\n')
    for filename, tags in zip(batch_paths, [line.split('Tags: ')[1] for line in output_lines if 'Tags: ' in line]):
        save_tags(filename, tags)

def main():
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']  # Add more extensions if needed
    image_files = [str(p) for p in Path(IMAGE_FOLDER).glob('*') if p.suffix.lower() in image_extensions]
    num_cores = int(0.8 * cpu_count())  # Use 80% of available CPU cores
    batch_size = 12  # Adjust batch size as needed

    print(f"Found {len(image_files)} image files in {IMAGE_FOLDER}")
    print(f"Using {num_cores} processes with batch size {batch_size}")

    args = []
    for start in range(0, len(image_files), batch_size * num_cores):
        batch_paths = image_files[start:start + batch_size * num_cores]
        for core in range(num_cores):
            core_paths = batch_paths[core * batch_size:(core + 1) * batch_size]
            if core_paths:
                args.append((core_paths, 'swinv2', 0.5, 0.8))
                print(f"Process {core + 1}: Processing {len(core_paths)} images")

    with Pool(processes=num_cores) as pool:
        pool.map(process_batch_wrapper, args)

    print("Tagging completed successfully!")

if __name__ == '__main__':
    main()