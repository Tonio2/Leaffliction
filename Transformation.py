import os
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
# 	# color histogram

def	transformation(img):
	transformation = {}

	img, path, filename = pcv.readimage(filename=img)
	transformation["gaussian_img"] = pcv.gaussian_blur(img, ksize=(7,7), sigma_x=0, sigma_y=None)

	
	for key, value in transformation.items():
		pcv.plot_image(value)


def main(img):
	try:
		# iterate through folders
		transformation(img)
		#render
	except Exception as error:
		print("Error:", error)
		exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python Transformation.py file")
        exit(1)
    main(sys.argv[1])
