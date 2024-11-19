import os
import sys
import shutil
from plantcv import learn
from plantcv import plantcv as pcv

def clean_directory(path):
        """Removes a directory if it exists."""
        if os.path.exists(path):
            print(f"Removing directory: {path}")
            try:
                os.system(f"rm -rf {path}")
            except Exception as e:
                print(f"Error removing directory {path}: {e}", file=sys.stderr)
                sys.exit(1)

def process_images(input_dir, output_images_dir="bayes_images", output_masks_dir="bayes_masks", output_file="output.txt"):
    """
    Processes images by copying them to a specified directory and creating masks.

    Args:
        input_dir (str): Directory containing input images.
        output_images_dir (str): Directory where processed images will be copied.
        output_masks_dir (str): Directory where masks will be saved.
        output_file (str): Path to the output file for the Naive Bayes classifier.
    """
    # Clean and recreate output directories
    clean_directory(output_images_dir)
    clean_directory(output_masks_dir)
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)

    try:
        subdirs = os.listdir(input_dir)
    except FileNotFoundError:
        print(f"Error: The directory '{input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    for subdir in subdirs:
        subdir_path = os.path.join(input_dir, subdir)
        if not os.path.isdir(subdir_path):
            print(f"Skipping non-directory: {subdir_path}", file=sys.stderr)
            continue

        for img in os.listdir(subdir_path):
            imgpath = os.path.join(subdir_path, img)
            print(f"Processing {imgpath}")
            new_imgpath = os.path.join(output_images_dir, f"{subdir}_{img}")

            # Copy the image directly instead of using pcv.print_image
            try:
                shutil.copy(imgpath, new_imgpath)
            except Exception as e:
                print(f"Error copying {imgpath} to {new_imgpath}: {e}", file=sys.stderr)
                continue

            # Process mask
            try:
                image, _, _ = pcv.readimage(filename=imgpath)
                b_channel = pcv.rgb2gray_lab(rgb_img=image, channel='b')
                bin_mask = pcv.threshold.otsu(gray_img=b_channel, object_type='light')
                clean_mask = pcv.fill_holes(bin_mask)
                pcv.print_image(clean_mask, os.path.join(output_masks_dir, f"{subdir}_{img}"))
            except Exception as e:
                print(f"Error processing mask for {imgpath}: {e}", file=sys.stderr)
                continue

    # Run Naive Bayes classifier
    try:
        learn.naive_bayes(output_images_dir, output_masks_dir, output_file)
        print(f"Naive Bayes classification output written to {output_file}")
    except Exception as e:
        print(f"Error running Naive Bayes classifier: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    input_directory = "images/Apple"

    process_images(input_directory)
