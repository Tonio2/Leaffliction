from plantcv import plantcv as pcv
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def select_best_channel(img):
    # Convert to LAB and CMYK
    b_channel = pcv.rgb2gray_lab(rgb_img=img, channel="b")
    y_channel = pcv.rgb2gray_cmyk(rgb_img=img, channel="y")

    # Analyze metrics for both channels
    metrics = {}
    for channel_name, channel_data in [("b", b_channel), ("y", y_channel)]:
        # Edge detection using Sobel
        edges = cv2.Sobel(channel_data, cv2.CV_64F, 1, 0, ksize=3)
        edge_variance = np.var(edges)

        # Intensity contrast (foreground-background separability)
        hist, _ = np.histogram(channel_data, bins=256, range=(0, 256))
        intensity_contrast = hist.max() - hist.min()

        metrics[channel_name] = {
            "edge_variance": edge_variance,
            "intensity_contrast": intensity_contrast,
        }

    # Select channel with highest combined score
    selected_channel = max(metrics, key=lambda k: metrics[k]["edge_variance"])

    return selected_channel, b_channel if selected_channel == "b" else y_channel


def gaussian_blur(img, channel="b"):
    a_gray = pcv.rgb2gray_hsv(rgb_img=img, channel="h")

    bin_mask = pcv.threshold.binary(gray_img=a_gray, threshold=110, object_type="dark")
    cleaned_mask = pcv.fill(bin_img=bin_mask, size=50)
    return cleaned_mask

def mask_objects(img, mask):
    masked = pcv.apply_mask(img=img, mask=mask, mask_color="white")
    return masked

def remove_black(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_img, 20, 255, cv2.THRESH_BINARY)
    masked = pcv.apply_mask(img=img, mask=mask, mask_color="white")
    return masked

def roi(img, mask):
    roi_image = img.copy()
    roi_image[mask != 5] = (0, 255, 0)
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
    
    # Get colorspqce
    colorspace_img = pcv.visualize.colorspaces(rgb_img=img)
    print_image(colorspace_img, "colorspace.jpg")
    
    cleaned_mask = gaussian_blur(img)
    print_image(cleaned_mask, "cleaned_mask.jpg")

    a_gray = pcv.rgb2gray_hsv(rgb_img=img, channel="h")
    analysis_image = pcv.analyze.grayscale(gray_img=a_gray, labeled_mask=cleaned_mask,
                                        n_labels=1, bins=100, 
                                        label="default")
    print_image(analysis_image, "analysis_image.png")

    selected_channel, channel_name = select_best_channel(img)
    print(f"Selected channel: {selected_channel}")
    print_image(channel_name, "channel_name.jpg")

    masked_img = mask_objects(img, cleaned_mask)
    print_image(masked_img, "masked_img.jpg")
    
    withouth_black = remove_black(masked_img)
    print_image(withouth_black, "without_black.jpg")

    roi_img = roi(masked_img, cleaned_mask)
    print_image(roi_img, "roi.jpg")

    shape_img = analyze_object(img, cleaned_mask)
    print_image(shape_img, "shape_img.jpg")

    pseudolandmarks_img = create_pseudolandmarks_image(img, cleaned_mask)
    print_image(pseudolandmarks_img, "pseudolandmarks_img.jpg")

    color_img = analyze_color(img, cleaned_mask)
    print(color_img)

if __name__ == '__main__':
    images = ["image1.JPG", "image2.JPG", "image3.JPG", "image4.JPG"]
    dirs = ["output", "output2", "output3", "output4"]
    for i, d in zip(images, dirs):
        if not os.path.exists(d):
            os.mkdir(d)
        main(i, d)