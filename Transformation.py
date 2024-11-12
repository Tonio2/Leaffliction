import sys
import matplotlib.pyplot as plt
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
    mask, masked_image = pcv.threshold.custom_range(img, lower_thresh=[10, 10, 10], upper_thresh=[255, 255, 100], channel='RGB')

    transformation["Original"] = img
    transformation["black_background_white_object"] = masked_image
    transformation["gaussian"] = pcv.gaussian_blur(masked_image, ksize=(7, 7), sigma_x=0, sigma_y=None)
    transformation["white_mask"] = pcv.apply_mask(img, mask=masked_image, mask_color='white'))
    # transformation["black_mask"] = pcv.apply_mask(img, mask=mask, mask_color='black')

    # Defining ROI 
    # -> roi = object area should be white an background should be black
    # -> device ???
    # device, roi, roi_hierarchy = pcv.define_roi(img=img, shape='rectangle', device=device, roi=transformation["black_background_white_object"], roi_input='binary', debug=None, adjust=False, x_adj=0, y_adj=0, w_adj=0, h_adj=0)
    # device, roi, roi_hierarchy = define_roi(img, 'rectangle', device, roi=None, roi_input='default', debug="print", adjust=True, x_adj=0, y_adj=0, w_adj=0, h_adj=-925)
	# transformation["Roi_objects"] = 

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
