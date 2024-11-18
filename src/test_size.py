import os
import sys
import cv2

def is_dir(dirname, d):
    return os.path.isdir(os.path.join(dirname, d))

def main(dirname):
    original_width, original_height = 0, 0
    for d in os.listdir(dirname):
        if not os.path.isdir(os.path.join(dirname, d)):
            continue
        for img in os.listdir(os.path.join(dirname, d)):
            # Get image dimensions
            img_path = os.path.join(dirname, d, img)
            image = cv2.imread(img_path)
            height, width, _ = image.shape
            if original_width == 0 and original_height == 0:
                original_width, original_height = width, height
            if width != original_width or height != original_height:
                print(f"Image {img} in directory {d} has different dimensions.")
                print(f"Original dimensions: {original_width}x{original_height}")
                print(f"Current dimensions: {width}x{height}")
                print("Exiting...")
                exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python Distribution.py dirname")
        exit(1)
    main(sys.argv[1])