import os
import cv2
import sys
import numpy as np
from Distribution import is_dir


def rotate(src: np.ndarray) -> np.ndarray:
    """ Rotates an image by 45 degrees clockwise """
    (h, w) = src.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    return cv2.warpAffine(src, rotation_matrix, (w, h))


def blur(src: np.ndarray) -> np.ndarray:
    """ Applies a Gaussian blur """
    return cv2.GaussianBlur(src, (9, 9), 1)


def contrast(src: np.ndarray) -> np.ndarray:
    """ Increases the contrast of the image """
    return cv2.convertScaleAbs(src, alpha=1.5, beta=10)


def flip(src: np.ndarray) -> np.ndarray:
    """ Flips the image horizontally (mirrors the image) """
    return cv2.flip(src, 1)


def illumination(src: np.ndarray) -> np.ndarray:
    """ Increases the image illumination """
    return cv2.convertScaleAbs(src, alpha=1, beta=50)


def projective_transformation(src: np.ndarray) -> np.ndarray:
    """ Applies a projective transformation to the image """
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


def scale(src: np.ndarray) -> np.ndarray:
    """ Scales the image while maintaining the aspect ratio """
    scale_factor = 1.2
    height, width = src.shape[:2]
    scaled_img = cv2.resize(src, None,
                            fx=scale_factor, fy=scale_factor,
                            interpolation=cv2.INTER_LINEAR)
    start_x = (scaled_img.shape[1] - width) // 2
    start_y = (scaled_img.shape[0] - height) // 2
    cropped_img = scaled_img[start_y:start_y + height, start_x:start_x + width]
    return cropped_img


# Careful: You should not resize images because the encoder needs a fixed size.
def zoom(src: np.ndarray) -> np.ndarray:
    """ Zooms the image """
    return cv2.resize(src, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)


def hue_adjustment(src: np.ndarray) -> np.ndarray:
    """ Adjusts the hue of the input image """
    return cv2.cvtColor(src, cv2. COLOR_BGR2HSV)


ft_map = {
    "Rotate": rotate,
    "Blur": blur,
    "Contrast": contrast,
    "Flip": flip,
    "Projective": projective_transformation,
    "Zoom": zoom,
    "Hue_Adjustment": hue_adjustment,
    "Illumination": illumination,
    "Scale": scale
}


params = ["Rotate", "Blur", "Contrast", "Flip", "Projective", "Scale"]


def augmentation(src: str) -> dict:
    """ Applies augmentations to the input image """
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


def setup(dir_path: str, output_dir: str = "$HOME/goinfre") -> str:
    """ Sets up the output directory for augmented images """
    output_dir = os.path.expandvars(output_dir)
    new_dir = os.path.join(output_dir, "augmented_directory")

    print("rm -rf " + new_dir)
    os.system("rm -rf " + new_dir)

    print("cp -r " + dir_path + " " + new_dir)
    os.system("cp -r " + dir_path + " " + new_dir)
    return new_dir


def main(dirname: str):
    """ Augments images in the specified directory """
    new_dir = setup(dirname, "./")

    dirs = [d for d in os.listdir(new_dir) if is_dir(new_dir, d)]
    dir_size = [len(os.listdir(os.path.join(new_dir, d))) for d in dirs]
    max_dir_size = 7 * (min(dir_size))

    print(f"Found {len(dirs)} directories:")
    for d, s in zip(dirs, dir_size):
        print(f"Directory {d} has {s} images.")
    print(f"Let's augment them to have {max_dir_size} images per directory.")

    for d in dirs:
        print(f"Augmenting {d}...")
        idx = 0
        img_dir = os.path.join(new_dir, d)
        img_list = os.listdir(img_dir)
        dir_size = len(img_list)

        while dir_size + 6 * idx <= max_dir_size and idx < dir_size:
            img = os.path.join(img_dir, img_list[idx])
            modifs = augmentation(img)
            for key, value in modifs.items():
                save_path = img.split(".JPG")[0] + key
                cv2.imwrite(save_path, value)
            idx += 1


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python Augmentation.py <image_path | dir_path>")
        sys.exit(1)

    if len(params) != 6:
        print("Please provide 6 augmentations in the params list.")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Augmentations are {', '.join(params)}.")
    print("You can change them in the params list.")
    if not os.path.exists(path):
        print(f"Path {path} does not exist.")
        sys.exit(1)

    if os.path.isfile(path):
        modifs = augmentation(path)
        for key, value in modifs.items():
            save_path = os.path.basename(path).split(".JPG")[0] + key
            cv2.imwrite(save_path, value)
    elif os.path.isdir(path):
        main(path)
