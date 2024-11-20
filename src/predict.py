import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from train import preprocess_img
from Transformation import bayes, mask_objects


def load_class_labels(label_file_path: str) -> list[str]:
    """ Load class labels from a file """
    if not os.path.exists(label_file_path):
        raise FileNotFoundError(f"Label file not found: {label_file_path}")
    with open(label_file_path, "r") as f:
        class_labels = [line.strip() for line in f.readlines()]
    return class_labels


def render(src, predicted_label, output_path):
    """ Render the the disease of a single image """
    text = "===    DL classification    ===\nClass predicted : " + predicted_label
    original_img = cv2.imread(src)
    if original_img is None:
        raise FileNotFoundError("Image not found or cannot be read.")

    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    mask = bayes(original_img)
    masked_img = mask_objects(original_img, mask)
    plt.figure(figsize=(10, 6), facecolor='black')

    combined_img = np.concatenate((original_img_rgb, masked_img), axis=1)
    blank_space = np.ones((10, combined_img.shape[1], 3), dtype=np.uint8)
    final_img = np.concatenate((combined_img, blank_space), axis=0)

    plt.imshow(final_img)
    plt.axis('off')
    plt.figtext(0.5, 0.01, text, ha='center', fontsize=12, color='white', weight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def predict(img_src, model_path, label_file):
    """ Predict the disease of a single image and render the corresponding class label """
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    class_labels = load_class_labels(label_file_path)

    # Preprocess the image
    processed_img = preprocess_img(img_src)
    processed_img = np.expand_dims(processed_img, axis=0)  # Add batch size

    # Predict
    predictions = model.predict(processed_img)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Get class labels
    predicted_label = class_labels[predicted_class]

    return predicted_label


def main(img_src: str,
                  model_path: str,
                  label_file_path: str,
                  output_path: str) -> None:
    """ Predict the disease of a single image and render the output """
    if os.path.isfile(img_src) is False:
        print("Error: {img_src} is not a file")
        exit(1)

    if os.path.isfile(model_path) is False:
        print("Error: {model_path} is not a file")
        exit(1)

    try:
        predicted_label = predict(img_src, model_path, label_file_path)
        render(img_src, predicted_label, output_path)

    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python predict.py <image_path> <model_path> \
            <label_file_path> <output_path>")
        exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2]
    label_file_path = sys.argv[3]
    output_path = sys.argv[4]

    main(image_path, model_path, label_file_path, output_path)
