import cv2
import numpy as np
import sys
import os

params = ["Rotate", "Blur", "Contrast", "Flip", "Projective", "Zoom"]


def rotate(src):
    (h, w) = src.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    return cv2.warpAffine(src, rotation_matrix, (w, h))


def blur(src):
    return cv2.GaussianBlur(src, (9, 9), 1)


def contrast(src):
    return cv2.convertScaleAbs(src, alpha=1.5, beta=10)


def flip(src):
    return cv2.flip(src, 1)


def illumination(src):
    return cv2.convertScaleAbs(src, alpha=1, beta=50)


def projective_transformation(src):
    (h, w) = src.shape[:2]
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
    return cv2.warpPerspective(src, matrix, output_size)


def zoom(src):
    return cv2.resize(src, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)


def hue_adjustment(src):
    return cv2.cvtColor(src, cv2. COLOR_BGR2HSV)


ft_map = {
    "Rotate": rotate,
    "Blur": blur,
    "Contrast": contrast,
    "Flip": flip,
    "Projective": projective_transformation,
    "Zoom": zoom,
    "Hue_Adjustment": hue_adjustment,
    "Illumination": illumination
}


def augmentation(src):
    modification = {}
    img = cv2.imread(src)
    if img is None:
        raise FileNotFoundError(f"Image {src} not found or cannot be read.")

    for ft in params:
        try:
            modification["_" + ft + ".JPG"] = ft_map[ft](img)
        except Exception as e:
            print(f"Error applying {ft} to {src}: {e}")

    return modification


def is_dir(dirname, d):
    return os.path.isdir(os.path.join(dirname, d))


def main(dirname):
    # Copy directory to new directory "augmented_directory"
    new_dir = "/mnt/nfs/homes/alabalet/goinfre/augmented_directory"
    print("cp -r " + dirname + " " + new_dir)
    os.system("cp -r " + dirname + " " + new_dir)

    dirs = [d for d in os.listdir(new_dir) if is_dir(new_dir, d)]
    print(dirs)
    dir_size = [len(os.listdir(os.path.join(new_dir, d))) for d in dirs]
    print(dir_size)
    max_dir_size = max(dir_size)

    for d in dirs:
        idx = 0
        img_list = os.listdir(os.path.join(new_dir, d))
        dir_size = len(img_list)
        while dir_size + 6 * idx <= max_dir_size:
            img = os.path.join(new_dir, d, img_list[idx])
            print(img)
            modifs = augmentation(img)
            for key, value in modifs.items():
                save_path = img.split(".JPG")[0] + key
                cv2.imwrite(save_path, value)
            idx += 1


if __name__ == '__main__':
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("Usage: python Augmentation.py <image_path> | " +
              "python Augmentation.py --dir <dirname>")
        sys.exit(1)
    if len(sys.argv) == 2:
        src = sys.argv[1]
        modifs = augmentation(src)
        for key, value in modifs.items():
            save_path = os.path.basename(src).split(".JPG")[0] + key
            cv2.imwrite(save_path, value)
    elif len(sys.argv) == 3:
        if sys.argv[1] != "--dir":
            print("Usage: python Augmentation.py <image_path> | " +
                  "python Augmentation.py --dir <dirname>")
            sys.exit(1)
        dirname = sys.argv[2]
        main(sys.argv[2])
