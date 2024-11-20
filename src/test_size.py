import os
import sys
import cv2


def main(dirname):
    """ """
    original_wdt, original_hght = 0, 0
    for d in os.listdir(dirname):
        if not os.path.isdir(os.path.join(dirname, d)):
            continue
        for img in os.listdir(os.path.join(dirname, d)):
            # Get image dimensions
            img_path = os.path.join(dirname, d, img)
            image = cv2.imread(img_path)
            height, width, _ = image.shape
            if original_wdt == 0 and original_hght == 0:
                original_wdt, original_hght = width, height
            if width != original_wdt or height != original_hght:
                print(f"Image {img} in directory {d} has different dimensions")
                print(f"Original dimensions: {original_wdt}x{original_hght}")
                print(f"Current dimensions: {width}x{height}")
                print("Exiting...")
                exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python Distribution.py dirname")
        exit(1)
    main(sys.argv[1])
