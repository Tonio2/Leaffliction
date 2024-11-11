import cv2
import numpy as np
import sys
import os


def augmentation(src, path):

    # (h, w) = img.shape[:2]
    # center = (w // 2, h // 2)
	# ajouter median blur
	modification = {}

	modification["_Median_Blur.JPG"] = cv2.medianBlur(src, 7)
	modification["_Blur.JPG"] = cv2.GaussianBlur(src, (7, 7), 10)
	modification["_Contrast.JPG"] = cv2.convertScaleAbs(src, alpha=0.5)
	modification["_Flip.JPG"] = cv2.flip(src, 270)
	modification["_Hue_Adjustment.JPG"] = cv2.cvtColor(src, cv2. COLOR_BGR2HSV)
	modification["_Illumination.JPG"] = cv2.convertScaleAbs(src, beta=50)
	modification["_Rotate.JPG"] = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE) 
	modification["_Zoom.JPG"] = cv2.resize(src, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)

	# save image 
	for key, value in modification.items():
		save_path = path.split(".JPG")[0] + key
		cv2.imwrite(save_path, value)


def transform(image_path):

    img = cv2.imread(image_path)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    # Rotate image
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated_image = cv2.warpAffine(img, rotation_matrix, (w, h))
    
    # Blur
    blurred_image = cv2.GaussianBlur(img, (15, 15), 0)
    
    # Flip  image
    flipped_image = cv2.flip(img, 1)
    
    # Enhance contrast
    enhanced_image = cv2.convertScaleAbs(img, alpha=1.5, beta=10)
    
    # Illuminate image
    illuminate_image = cv2.convertScaleAbs(img, alpha=1, beta=50)
    
    # Projective transformation
    src_points = np.float32([
        [0, 0],           # Top-left corner
        [w, 0],           # Top-right corner
        [0, h],           # Bottom-left corner
        [w, h]            # Bottom-right corner
    ])

    dst_points = np.float32([
        [w * 0.2, 0],
        [w * 0.8, 0],
        [0, h],
        [w, h]
    ])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    output_size = (h, w)
    warped_image = cv2.warpPerspective(img, matrix, output_size)
    
    return rotated_image, blurred_image, flipped_image, enhanced_image, illuminate_image, warped_image


def is_dir(dirname, d):
    return os.path.isdir(os.path.join(dirname, d))

def main(dirname):
    # Copy directory to new directory "augmented_directory"
    new_dir = "augmented_directory"
    os.system("cp -r " + dirname + " " + new_dir)
    
    dirs = [d for d in os.listdir(new_dir) if is_dir(new_dir, d)]
    dir_size = [len(os.listdir(os.path.join(new_dir, d))) for d in dirs]
    max_dir_size = max(dir_size)
    
    for d in dirs:
        idx = 0
        img_list = os.listdir(os.path.join(new_dir, d))
        dir_size = len(img_list)
        while dir_size + 6 * idx <= max_dir_size:
            img = os.path.join(new_dir, d, img_list[idx])
            basename = os.path.join(new_dir, d, img_list[idx].split(".")[0])
            r, b, f, e, i, w = transform(img)
            cv2.imwrite(basename + "_Rotate.JPG", r)
            cv2.imwrite(basename + "_Blur.JPG", b)
            cv2.imwrite(basename + "_Flip.JPG", f)
            cv2.imwrite(basename + "_Enhance.JPG", e)
            cv2.imwrite(basename + "_Illuminate.JPG", i)
            cv2.imwrite(basename + "_Project.JPG", w)
            idx += 1

if __name__ == '__main__':
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("Usage: python Augmentation.py <image_path> | python Augmentation.py --dir <dirname>")
        sys.exit(1)
    if len(sys.argv) == 2:
        r, b, f, e, i, w = transform(sys.argv[1])
        basename = os.path.basename(sys.argv[1]).split(".")[0]
        cv2.imwrite(basename + "_Rotate.JPG", r)
        cv2.imwrite(basename + "_Blur.JPG", b)
        cv2.imwrite(basename + "_Flip.JPG", f)
        cv2.imwrite(basename + "_Enhance.JPG", e)
        cv2.imwrite(basename + "_Illuminate.JPG", i)
        cv2.imwrite(basename + "_Project.JPG", w)
    elif len(sys.argv) == 3:
        if sys.argv[1] != "--dir":
            print("Usage: python Augmentation.py <image_path> | python Augmentation.py --dir <dirname>")
            sys.exit(1)
        dirname = sys.argv[2]
        main(sys.argv[2])