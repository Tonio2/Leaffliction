from plantcv import plantcv as pcv
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def gaussian_blur(img):
    a_gray = pcv.rgb2gray_lab(rgb_img=img, channel="a")
    bin_mask = pcv.threshold.otsu(gray_img=a_gray, object_type="dark")
    cleaned_mask = pcv.fill(bin_img=bin_mask, size=50)
    return cleaned_mask

def mask_objects(img, mask):
    masked = pcv.apply_mask(img=img, mask=mask, mask_color="white")
    return masked

def roi(img, mask):
    roi_image = img.copy()
    roi_image[mask != 0] = (0, 255, 0)
    x, y, w, h = cv2.boundingRect(mask)
    roi_image = cv2.rectangle(roi_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return roi_image

def analyze_object(img, mask):
    shape_img = pcv.analyze.size(img=img, labeled_mask=mask)
    return shape_img

def draw_circle(image, center, radius, color):
    """Draw a circle on the image at the specified center with given radius and color."""
    x, y = np.ogrid[:image.shape[0], :image.shape[1]]
    mask = (x - center[1]) ** 2 + (y - center[0]) ** 2 <= radius ** 2
    image[mask] = color
    return image


def create_pseudolandmarks_image(image, mask):
    # Create a pseudolandmarks image
    pseudolandmarks = image.copy()
    top_x, bottom_x, center_v_x = pcv.homology.x_axis_pseudolandmarks(
        img=pseudolandmarks, mask=mask, label='default'
    )
    colors = [(0, 0, 255), (255, 0, 255), (255, 0, 0)]
    for points, color in zip([top_x, bottom_x, center_v_x], colors):
        for point in points:
            center = (point[0][0], point[0][1])
            draw_circle(pseudolandmarks, center, radius=5, color=color)
    return pseudolandmarks

def analyze_color(img, mask):
    # Analyze color
    color_img = pcv.analyze.color(rgb_img=img, labeled_mask=mask, colorspaces='all')
    hue_data = pcv.outputs.observations['default_1']['hue_frequencies']['value']
    saturation_data = pcv.outputs.observations['default_1']['saturation_frequencies']['value']
    value_data = pcv.outputs.observations['default_1']['value_frequencies']['value']
    lightness_data = pcv.outputs.observations['default_1']['lightness_frequencies']['value']
    blue_data = pcv.outputs.observations['default_1']['blue_frequencies']['value']
    green_data = pcv.outputs.observations['default_1']['green_frequencies']['value']
    red_data = pcv.outputs.observations['default_1']['red_frequencies']['value']
    green_magenta_data = pcv.outputs.observations['default_1']['green-magenta_frequencies']['value']
    blue_yellow_data = pcv.outputs.observations['default_1']['blue-yellow_frequencies']['value']

    # Plot the histogram for each color channel
    plt.figure(figsize=(10, 6))
    plt.plot(hue_data, label="hue", color="magenta")
    plt.plot(saturation_data, label="saturation", color="cyan")
    plt.plot(value_data, label="value", color="blue")
    plt.plot(lightness_data, label="lightness", color="gray")
    plt.plot(blue_data, label="blue", color="blue")
    plt.plot(green_data, label="green", color="green")
    plt.plot(red_data, label="red", color="red")
    plt.plot(green_magenta_data, label="green-magenta", color="purple")
    plt.plot(blue_yellow_data, label="blue-yellow", color="yellow")

    plt.xlabel("Pixel intensity")
    plt.ylabel("Proportion of pixels (%)")
    plt.legend(title="Color Channel")
    plt.title("Color Channel Histogram")
    plt.savefig(os.path.join("output", "color_histogram.png"))


def main(img_path, output):
    def print_image(img, filename):
        # Print image
        pcv.print_image(img, os.path.join(output, filename))

    # Delete all files in output directory
    for file in os.listdir(output):
        os.remove(os.path.join(output, file))

    # Read image
    img, path, filename = pcv.readimage(filename=img_path)

    cleaned_mask = gaussian_blur(img)
    print_image(cleaned_mask, "cleaned_mask.jpg")

    masked_img = mask_objects(img, cleaned_mask)
    print_image(masked_img, "masked_img.jpg")

    roi_img = roi(masked_img, cleaned_mask)
    print_image(roi_img, "roi.jpg")

    shape_img = analyze_object(img, cleaned_mask)
    print_image(shape_img, "shape_img.jpg")

    pseudolandmarks_img = create_pseudolandmarks_image(img, cleaned_mask)
    print_image(pseudolandmarks_img, "pseudolandmarks_img.jpg")

    color_img = analyze_color(img, cleaned_mask)
    print(color_img)


main("image.JPG", "output")