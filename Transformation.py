import sys
from plantcv import plantcv as pcv

# Pseudo code
# check arg
# protect error
# retrieve data
# apply modification


# testing on a single plant first

# def render():
#     # color histogram

def transformation(src):
    transformation = {}

    img, path, filename = pcv.readimage(filename=src)
    mask, masked_image = pcv.threshold.custom_range(img, lower_thresh=[10, 10, 10], upper_thresh=[100, 255, 100], channel='RGB')

    transformation["Original"] = img
    transformation["black_and_white"] = mask
    transformation["gaussian"] = pcv.gaussian_blur(mask, ksize=(7, 7), sigma_x=0, sigma_y=None)
    transformation["white_mask"] = pcv.apply_mask(img, mask=mask, mask_color='white')

    for key, value in transformation.items():
        pcv.plot_image(value)


def main(img):
    try:
        # iterate through folders
        transformation(img)
        # render
    except Exception as error:
        print("Error:", error)
        exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python Transformation.py file")
        exit(1)
    main(sys.argv[1])
