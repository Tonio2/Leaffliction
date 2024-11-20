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


def analyze_color(
    img: np.ndarray,
    mask: np.ndarray,
    output: str,
    basename: str
) -> None:
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


def bayes(img: np.ndarray, pdf_file: str = "./output.txt") -> np.ndarray:
    """ Generate a mask using Bayesian classification """
    mask = pcv.naive_bayes_classifier(rgb_img=img, pdf_file=pdf_file)
    plant_mask = mask['plant']
    plant_mask = pcv.fill_holes(plant_mask)

    return plant_mask


def process_image(
    img_path: str,
    output_dir: str,
    fruit: str,
    plot: bool = False
) -> None:
    """ Process a single image with the specified transformation """
    try:
        img, _, _ = pcv.readimage(filename=img_path)
        bname = os.path.basename(img_path).split(".")[0]

        images = [(img, "Original Image")]
        mask = gaussian_blur(img)
        images.append((mask, "Mask"))

        result = mask_objects(img, mask)
        images.append((result, "Masked Image"))

        result = roi(img, mask)
        images.append((result, "ROI Image"))

        result = analyze_object(img, mask)
        images.append((result, "Analysis"))

        result = create_pseudolandmarks_image(img, mask)
        images.append((result, "Pseudolandmarks"))

        analyze_color(img, mask, output_dir, bname)

        mask = bayes(img, pdf_file=f"{fruit}.txt")
        result = mask_objects(img, mask)
        pcv.print_image(result, os.path.join(output_dir,
                                             f"{bname}_bayes.png"))

        if not plot:
            pcv.print_image(mask, os.path.join(output_dir,
                                               f"{bname}_mask.png"))
            pcv.print_image(images[2][0],
                            os.path.join(output_dir, f"{bname}_masked.png"))
            pcv.print_image(images[3][0],
                            os.path.join(output_dir, f"{bname}_roi.png"))
            pcv.print_image(images[4][0],
                            os.path.join(output_dir, f"{bname}_analysis.png"))
            pcv.print_image(images[5][0],
                            os.path.join(output_dir, f"{bname}_landmarks.png"))
        else:
            plt.figure(figsize=(15, 10))
            for i, (image, title) in enumerate(images, 1):
                plt.subplot(2, 3, i)
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.title(title)
                plt.axis('off')
            plt.tight_layout()
            plt.show()
            plt.savefig(os.path.join(output_dir, f"{bname}_results.png"))
    except Exception as e:
        print(f"Error processing image: {e}")
        raise


# CLI setup
def main():
    parser = argparse.ArgumentParser(description="Image Transformation Script")
    parser.add_argument("input", help="Path to the input image or directory.")
    parser.add_argument("output", help="Directory to save results.")
    parser.add_argument("fruit", help="Fruit to analyze.")
    args = parser.parse_args()

    try:
        os.system("rm -rf " + args.output)
        os.makedirs(args.output, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory: {e}")
        raise

    if os.path.isfile(args.input):
        process_image(args.input, args.output, args.fruit, plot=True)
    elif os.path.isdir(args.input):
        for file in os.listdir(args.input):
            if file.lower().endswith(("jpg", "jpeg", "png")):
                process_image(os.path.join(args.input, file),
                              args.output, args.fruit)
    else:
        print("Invalid input path. Provide an image or directory.")


if __name__ == "__main__":
    main()
