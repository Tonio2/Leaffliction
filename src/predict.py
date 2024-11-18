import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv

     # Hori = np.concatenate((img1, img2), axis=1)
     # cv2.imshow('Predict', Hori) 
    # # plt.imshow(img, cmap='gray')
    # cv2.waitKey(0) 
    # plt.show()

def render(img1, img2, state):
    text = "Class predicted : " + state
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    combined_img = np.concatenate((img1_rgb, img2_rgb), axis=1)

    plt.figure(figsize=(10, 5))
    plt.imshow(combined_img)

    plt.axis('off')
    plt.figtext(0.5, 0.01, text, ha='center', fontsize=12, color='black', weight='bold')
    plt.tight_layout()
    plt.show()


def main(src):
    # check if is file
    if os.path.isfile(src) == False:
        print("Not a file")
        exit(1)
    print("ALL GOOD WITH THE IMAGE")

    try:
        img1 = cv2.imread(src)
        if img1 is None:
            raise FileNotFoundError(f"Image not found or cannot be read.")
        print("IN TRY: ALL GOOD WITH THE IMAGE")

        # should apply tranfo 
        img2 = cv2.imread(src)
        if img2 is None:
            raise FileNotFoundError(f"Image not found or cannot be read.")

        # send to train to get the result
        state = "Healthy"

        render(img1, img2, state)

    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python predict.py 'path_image'")
        exit(1)
    main(sys.argv[1])
