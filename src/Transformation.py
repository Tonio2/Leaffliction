import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv


def draw_circle(image: np.ndarray,
                center: tuple[int, int],
                radius: int,
                color: tuple[int, int, int]) -> np.ndarray:
    """Draw a filled circle on the image."""
    x, y = np.ogrid[:image.shape[0], :image.shape[1]]
    mask = (x - center[1]) ** 2 + (y - center[0]) ** 2 <= radius ** 2
    image[mask] = color
    return image


# Transformation functions
def gaussian_blur(img: np.ndarray, channel: str = "b") -> np.ndarray:
    """Applies Gaussian blur based on specified channel."""
    try:
        if channel in ["l", "a", "b"]:
            a_gray = pcv.rgb2gray_lab(rgb_img=img, channel=channel)
        elif channel in ["h", "s", "v"]:
            a_gray = pcv.rgb2gray_hsv(rgb_img=img, channel=channel)
        elif channel in ["c", "m", "y", "k"]:
            a_gray = pcv.rgb2gray_cmyk(rgb_img=img, channel=channel)
        else:
            raise ValueError("Invalid channel specified.")
        bin_mask = pcv.threshold.otsu(gray_img=a_gray, object_type="light")
        cleaned_mask = pcv.fill_holes(bin_mask)
        return cleaned_mask
    except Exception as e:
        print(f"Error in gaussian_blur: {e}")
        raise


def mask_objects(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """ Apply a binary mask to an image """
    return pcv.apply_mask(img=img, mask=mask, mask_color="white")


def remove_black(img: np.ndarray) -> np.ndarray:
    """ Remove black regions from an image """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_img, 20, 255, cv2.THRESH_BINARY)
    return pcv.apply_mask(img=img, mask=mask, mask_color="white")


def roi(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """ Draw a rectangle around the region of interest (ROI) """
    roi_image = img.copy()
    roi_image[mask != 0] = (0, 255, 0)
    x, y, w, h = cv2.boundingRect(mask)
    return cv2.rectangle(roi_image, (x, y), (x + w, y + h), (255, 0, 0), 2)


def analyze_object(img: np.ndarray, mask: np.ndarray) -> dict:
    """ Analyze size and shape characteristics of an object """
    return pcv.analyze.size(img=img, labeled_mask=mask)


def create_pseudolandmarks_image(image: np.ndarray,
                                 mask: np.ndarray) -> np.ndarray:
    """ Create an image with pseudolandmarks visualized """
    pseudolandmarks = image.copy()
    top_x, bottom_x, center_v_x = pcv.homology.x_axis_pseudolandmarks(
        img=pseudolandmarks, mask=mask, label='default'
    )
    colors = [(0, 0, 255), (255, 0, 255), (255, 0, 0)]
    for points, color in zip([top_x, bottom_x, center_v_x], colors):
        for point in points:
            center = (point[0][0], point[0][1])
            pseudolandmarks = draw_circle(pseudolandmarks, center, 5, color)
    return pseudolandmarks


def analyze_color(img: np.ndarray, mask: np.ndarray, output: str, basename: str) -> None:
    """ Analyze color channels and save histograms """
    try:
        pcv.analyze.color(rgb_img=img, labeled_mask=mask, colorspaces='all')
        color_data = pcv.outputs.observations['default_1']
        histograms = {
            channel: color_data[f'{channel}_frequencies']['value']
            for channel in ['hue', 'saturation', 'value', 'lightness', 'blue',
                            'green', 'red', 'green-magenta', 'blue-yellow']
        }

        plt.figure(figsize=(10, 6))
        for channel, data in histograms.items():
            plt.plot(data, label=channel)
        plt.legend()
        plt.title("Color Channel Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Proportion")
        plt.savefig(os.path.join(output, f"{basename}_color_histogram.png"))
        plt.close()
    except Exception as e:
        print(f"Error in analyze_color: {e}")
        raise


def bayes(img: np.ndarray) -> np.ndarray:
    """ Generate a mask using Bayesian classification """
    mask = pcv.naive_bayes_classifier(rgb_img=img, pdf_file="./output.txt")
    plant_mask = mask['plant']
    plant_mask = pcv.fill_holes(plant_mask)

    return plant_mask


def process_image(img_path: str, output_dir: str) -> None:
    """ Process a single image with the specified transformation """
    try:
        img, _, _ = pcv.readimage(filename=img_path)
        basename = os.path.basename(img_path).split(".")[0]

        mask = gaussian_blur(img)
        pcv.print_image(mask, os.path.join(output_dir, f"{basename}_mask.png"))
        result = mask_objects(img, mask)
        pcv.print_image(result, os.path.join(output_dir, f"{basename}_masked.png"))
        result = roi(img, mask)
        pcv.print_image(result, os.path.join(output_dir, f"{basename}_roi.png"))
        result = analyze_object(img, mask)
        pcv.print_image(result, os.path.join(output_dir, f"{basename}_analysis.png"))
        result = create_pseudolandmarks_image(img, mask)
        pcv.print_image(result, os.path.join(output_dir, f"{basename}_landmarks.png"))
        analyze_color(img, mask, output_dir, basename)
        mask = bayes(img)
        result = mask_objects(img, mask)
        pcv.print_image(result, os.path.join(output_dir, f"{basename}_bayes.png"))
        print(f"Transformation results saved in {output_dir}.")
    except Exception as e:
        print(f"Error processing image: {e}")
        raise


# CLI setup
def main():
    parser = argparse.ArgumentParser(description="Image Transformation Script")
    parser.add_argument("input", help="Path to the input image or directory.")
    parser.add_argument("output", help="Directory to save results.")
    args = parser.parse_args()

    try:
        os.system("rm -rf " + args.output)
        os.makedirs(args.output, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory: {e}")
        raise

    if os.path.isfile(args.input):
        process_image(args.input, args.output)
    elif os.path.isdir(args.input):
        for file in os.listdir(args.input):
            if file.lower().endswith(("jpg", "jpeg", "png")):
                process_image(os.path.join(args.input, file),
                              args.output)
    else:
        print("Invalid input path. Provide an image or directory.")


if __name__ == "__main__":
    main()
