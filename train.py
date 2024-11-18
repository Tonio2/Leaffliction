import os
import sys
import cv2
import keras
import tensorflow
import numpy as np
import matplotlib.pyplot
from plantcv import plantcv as pcv
from sklearn.preprocessing import LabelEncoder
from transfo2 import gaussian_blur, mask_objects, remove_black
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


c_red = '\033[91m'
c_green = '\033[92m'
c_blue = '\033[94m'
c_yellow = '\033[93m'
c_reset = '\033[0m'


def count(set):
    """ Count the number of images in each category """
    counts = {
        'Apple_healthy': 0,
        'Apple_scab': 0,
        'Apple_Black_rot': 0,
        'Apple_rust': 0,        
    }
    total = 0
    for img, label in set:
        counts[label] += 1
        total += 1
    print(counts)
    return total


def normalize_img(array):
    """ Normalize the images """
    img_array = array.astype('float32') / 255.0
    return img_array


def preprocess_img(img):
    """ Preprocess the images """
    list_img_array = []
    return img


def dataset(dirname):
    """ """
    # Load the images
    dirs = [d for d in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, d))]
    array = []
    for d in dirs:
        for img in os.listdir(os.path.join(dirname, d)):
            array.append((os.path.join(dirname, d, img), d))

    total = count(array)
    print(c_red, f"Total images: {total}", c_reset)

    list_img = []
    img_size = 64

    # Transformation
    print(c_blue, "Transforming images...", c_reset)
    for img_path, label in array:
        img = cv2.imread(img_path)

        mask = gaussian_blur(img)
        img = mask_objects(img, mask)
        img = remove_black(img)

        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        list_img.append((img, label))
    print(c_green, "Transformation complete.", c_reset)

    # Normalize
    print(c_blue, "Normalizing images...", c_reset)
    images = np.array([i[0] for i in list_img], dtype="float32") / 255.0
    labels = [i[1] for i in list_img]
    print(c_yellow, images.shape, c_reset)
    print(c_green, "Normalization complete.", c_reset)

    # Labelize
    le = LabelEncoder()
    le.fit(labels)
    labels_encoder = le.fit_transform(labels)

    for i in range(0, len(labels), 250):
        print(f"Original: {labels[i]}, Encoded: {labels_encoder[i]}")

    # Split
    X_train, X_test, Y_train, Y_test = train_test_split(images, labels_encoder, train_size=0.85, random_state=42)
    Y_train = to_categorical(Y_train, num_classes=4)
    Y_test = to_categorical(Y_test, num_classes=4)
    print(c_yellow, X_train.shape, c_yellow, X_test.shape, c_blue, Y_train.shape, c_green, Y_test.shape, c_reset)

    # Define model
    learning_rate_decay = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=2)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True)

    model = keras.models.Sequential([
        keras.Input(shape=(64,64, 3)),

        keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Dropout(0.3),

        keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Dropout(0.3),

        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(4, activation="softmax")
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # Train model
    model.fit(x = X_train,
            y = Y_train,
            epochs = 50,
            callbacks = [learning_rate_decay, early_stopping],
            validation_split = 0.15)

    # Save model
    model.save("model.keras")

    # Evaluate model
    model.evaluate(x = X_test, y = Y_test)


def main(dirname):
    dataset(dirname)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python train.py dirname")
        exit(1)
    main(sys.argv[1])
