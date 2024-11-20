import os
import sys
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
from Augmentation import is_dir
from sklearn.preprocessing import LabelEncoder
from Transformation import bayes, mask_objects
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


# Masque les logs d'information et les warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

c_red = '\033[91m'
c_green = '\033[92m'
c_blue = '\033[94m'
cyel = '\033[93m'
cres = '\033[0m'


def normalize_img(images: list[np.ndarray]) -> np.ndarray:
    """ Normalize image pixel values to the range [0, 1] """
    return np.array(images, dtype="float32") / 255.0


def preprocess_img(image_path: str, img_size: int = 64) -> np.ndarray:
    """ Preprocess a single image """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image at {image_path}")

    # Apply custom transformations
    # mask = bayes(img)
    # img = mask_objects(img, mask)

    # Resize the image
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return img


def load_dataset(dirname: str) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """ Load and preprocess images from a dataset directory """
    print(c_blue, "Loading dataset...", cres)
    dirs = [d for d in os.listdir(dirname)
            if os.path.isdir(os.path.join(dirname, d))]
    image_paths_labels = []
    for d in dirs:
        for img in os.listdir(os.path.join(dirname, d)):
            image_paths_labels.append((os.path.join(dirname, d, img), d))

    processed_images = []
    labels = []

    # Transformation
    print(c_blue, "Transforming images...", cres)

    for img_path, label in image_paths_labels:
        img = preprocess_img(img_path)
        processed_images.append(img)
        labels.append(label)

    print(c_green, "Transformation complete.", cres)

    # Normalize
    images = normalize_img(processed_images)
    print(cyel, images.shape, cres)

    # Labelize
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    print(c_green, "Dataset loaded successfully.", cres)
    return images, encoded_labels, label_encoder


def build_model(input_shape: tuple[int, int, int], nb_cls: int) -> keras.Model:
    """ Build and compile a CNN model for image classification """
    model = keras.models.Sequential([
        keras.Input(shape=input_shape),

        keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(0.3),

        keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(0.3),

        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(nb_cls, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_model(model: keras.Model,
                X_train: np.ndarray,
                Y_train: np.ndarray) -> keras.Model:
    """ Train the CNN model """
    print(c_blue, "Starting training...", cres)

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                          factor=0.1,
                                          patience=2),
        keras.callbacks.EarlyStopping(monitor='val_loss',
                                      patience=8,
                                      restore_best_weights=True)
    ]

    model.fit(
        x=X_train, y=Y_train,
        epochs=50,
        callbacks=callbacks,
        validation_split=0.15)

    print(c_green, "Training complete.", cres)
    return model


def extract_fruit(labels: list[str]) -> str:
    """ Extract fruit name """
    fruit = list(set(d.split("_")[0] for d in labels))
    return fruit[0]


def evaluate_model(model: keras.Model,
                   X_test: np.ndarray,
                   Y_test: np.ndarray,
                   class_labels: list[str]) -> None:
    """ Evaluate the model and display metrics """
    print(c_blue, "Evaluating model...", cres)
    fruit = extract_fruit(class_labels)

    test_loss, test_accuracy = model.evaluate(X_test, Y_test)
    print(c_green, f"Test Loss: {test_loss:.4f}, \
                    Test Accuracy: {test_accuracy:.4f}", cres)

    # Predictions
    Y_pred_prob = model.predict(X_test)
    Y_pred = np.argmax(Y_pred_prob, axis=1)
    Y_test_classes = np.argmax(Y_test, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(Y_test_classes, Y_pred)
    ConfusionMatrixDisplay(cm, display_labels=class_labels).plot()
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig("results/" + fruit + "_conf_matrix.png")
    plt.show()

    # Classification Report
    print("\n" + classification_report(Y_test_classes, Y_pred, class_labels))

    # ROC Curve and AUC
    print(c_blue, "Generating ROC Curve...", cres)
    plt.figure()
    for i, label in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(Y_test[:, i], Y_pred_prob[:, i])
        auc = roc_auc_score(Y_test[:, i], Y_pred_prob[:, i])
        plt.plot(fpr, tpr, label=f"{label} (AUC: {auc:.2f})")

    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("results/" + fruit + "_ROC_curve.png")
    plt.show()


def main_pipeline(dataset_path: str) -> None:
    """Full pipeline: data loading, preprocess, model training, evaluation"""
    dirnames = [d for d in os.listdir(dataset_path) if is_dir(dataset_path, d)]
    name = f"results/{extract_fruit(dirnames)}.keras"

    # load & process the dataset
    images, labels, label_encoder = load_dataset(dataset_path)
    class_labels = label_encoder.classes_

    label_file_path = f"results/{extract_fruit(class_labels)}_labels.txt"
    with open(label_file_path, "w") as f:
        for label in class_labels:
            f.write(label + "\n")
    print(c_green, f"Class labels saved to {label_file_path}.", cres)

    # split the dataset
    print(c_blue, "Splitting dataset...", cres)
    X_train, X_test, Y_train, Y_test = train_test_split(images, labels,
                                                        train_size=0.85,
                                                        random_state=42)
    Y_train = to_categorical(Y_train, num_classes=len(class_labels))
    Y_test = to_categorical(Y_test, num_classes=len(class_labels))

    print(cyel, X_train.shape, X_test.shape, Y_train.shape, Y_test.shape, cres)

    # build & train the model
    model = build_model(X_train.shape[1:], len(class_labels))
    model = train_model(model, X_train, Y_train)

    # Save the model
    model.save(name)
    print(c_green, "Model saved.", cres)

    # Evaluate the model
    evaluate_model(model, X_test, Y_test, class_labels)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python train.py dirname")
        exit(1)

    dataset_path = sys.argv[1]
    main_pipeline(dataset_path)
