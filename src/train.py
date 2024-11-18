import os
import sys
import cv2
import keras
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from transfo2 import gaussian_blur, mask_objects, remove_black
from sklearn.metrics import roc_auc_score ,roc_curve, confusion_matrix, ConfusionMatrixDisplay, classification_report


c_red = '\033[91m'
c_green = '\033[92m'
c_blue = '\033[94m'
c_yellow = '\033[93m'
c_reset = '\033[0m'


def normalize_img(images):
    """ Normalize image pixel values to the range [0, 1] """
    return np.array(images, dtype="float32") / 255.0


def preprocess_img(image_path, img_size=64):
    """ Preprocess a single image """
    img = cv2.imread(image_path)

    # Apply custom transformations
    mask = gaussian_blur(img)
    img = mask_objects(img, mask)
    img = remove_black(img)

    # Resize the image
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return img


def load_dataset(dirname, img_size = 64):
    """ Load and preprocess images from a dataset directory """
    print(c_blue, "Loading dataset...", c_reset)
    dirs = [d for d in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, d))]
    image_paths_labels = []
    for d in dirs:
        for img in os.listdir(os.path.join(dirname, d)):
            image_paths_labels.append((os.path.join(dirname, d, img), d))

    processed_images = []
    labels = []

    # Transformation
    print(c_blue, "Transforming images...", c_reset)

    for img_path, label in image_paths_labels:
        img = preprocess_img(img_path, img_size = img_size)
        processed_images.append(img)
        labels.append(label)

    print(c_green, "Transformation complete.", c_reset)

    # Normalize
    images = normalize_img(processed_images)
    print(c_yellow, images.shape, c_reset)

    # Labelize
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    for i in range(0, len(labels), 250):
        print(f"Original: {labels[i]}, Encoded: {encoded_labels[i]}")

    print(c_green, "Dataset loaded successfully.", c_reset)
    return images, encoded_labels, label_encoder


def build_model(input_shape, num_classes):
    """ Build and compile a CNN model for image classification """
    model = keras.models.Sequential([
        keras.Input(shape=input_shape),

        keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Dropout(0.3),

        keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Dropout(0.3),

        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_model(model, X_train, Y_train):
    """ Train the CNN model """
    print(c_blue, "Starting training...", c_reset)

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    ]

    model.fit(
        x=X_train, y=Y_train,
        epochs = 50,
        callbacks = callbacks,
        validation_split = 0.15)
    
    print(c_green, "Training complete.", c_reset)
    return model


def evaluate_model(model, X_test, Y_test, class_labels):
    """ Evaluate the model and display metrics """
    print(c_blue, "Evaluating model...", c_reset)

    test_loss, test_accuracy = model.evaluate(X_test, Y_test)
    print(c_green, f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}", c_reset)

    # Predictions
    Y_pred_prob = model.predict(X_test)
    Y_pred = np.argmax(Y_pred_prob, axis=1)
    Y_test_classes = np.argmax(Y_test, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(Y_test_classes, Y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels).plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("conf_matrix.png")
    plt.show()

    # Classification Report
    print("\n" + classification_report(Y_test_classes, Y_pred, target_names=class_labels))

    # ROC Curve and AUC
    print(c_blue, "Generating ROC Curve...", c_reset)
    plt.figure()
    fpr, tpr, _ = roc_curve(Y_test, Y_pred_prob)
    plt.plot(fpr, tpr)

    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig("ROC_curve.png")
    plt.show()
    print(c_yellow, f"AUC f{roc_auc_score(Y_test, Y_pred_prob, average=None):.3f}" )


def main_pipeline(dataset_path):
    """ Full pipeline: data loading, preprocessing, model training, and evaluation """
    # load & process the dataset
    images, labels, label_encoder = load_dataset(dataset_path, img_size=64)
    class_labels = label_encoder.classes_

    # split the dataset
    print("Splitting dataset...")
    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, train_size=0.85, random_state=42)
    Y_train = to_categorical(Y_train, num_classes=len(class_labels))
    Y_test = to_categorical(Y_test, num_classes=len(class_labels))

    print(c_yellow, X_train.shape, c_yellow, X_test.shape, c_blue, Y_train.shape, c_green, Y_test.shape, c_reset)

    # build & train the model
    model = build_model(input_shape=X_train.shape[1:], num_classes=len(class_labels) )
    model = train_model(model, X_train, Y_train)

    # Save the model
    model.save("model.keras")
    print(c_green, "Model saved.", c_reset)

    # Evaluate the model
    evaluate_model(model, X_test, Y_test, class_labels)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python train.py dirname")
        exit(1)

    dataset_path = sys.argv[1]
    main_pipeline(dataset_path)
