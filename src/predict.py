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


def predict_image(image_path: str,
                  model_path: str,
                  label_file_path: str,
                  output_path: str) -> None:
    """ Predict the disease of a single image and render the output """
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    class_labels = load_class_labels(label_file_path)

    # Preprocess the image
    processed_img = preprocess_img(image_path)
    processed_img = np.expand_dims(processed_img, axis=0)  # Add batch size

    # Predict
    predictions = model.predict(processed_img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    # confidence = predictions[0][predicted_class]

    # Get class labels
    predicted_label = class_labels[predicted_class]

    # Load the original image
    original_img = cv2.imread(image_path)
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # Visualize preprocessing steps (e.g., mask application)
    mask = bayes(original_img)
    masked_img = mask_objects(original_img, mask)

    # Plot and save the output
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(original_img_rgb)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(masked_img)
    ax[1].set_title("Processed Image")
    ax[1].axis("off")

    plt.suptitle(f"DL Classification\nClass Predicted: {predicted_label}",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python predict.py <image_path> <model_path> \
            <label_file_path> <output_path>")
        exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2]
    label_file_path = sys.argv[3]
    output_path = sys.argv[4]

    predict_image(image_path, model_path, label_file_path, output_path)
