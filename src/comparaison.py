import os
import sys
import cv2
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.applications import ResNet50
from train import load_dataset


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

c_red = '\033[91m'
c_green = '\033[92m'
c_blue = '\033[94m'
cyel = '\033[93m'
cres = '\033[0m'


def vgg_model():
    encodeur = VGG16(weights="imagenet", include_top=False, input_shape=(64, 64, 3))
    encodeur.trainable = False

    model_vgg16_tl = keras.Sequential(
        [
            encodeur,
            keras.layers.Flatten(),
            keras.layers.Dense(1500, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(150, activation='relu'),
            keras.layers.Dense(4, "softmax")
        ]
    )
    model_vgg16_tl.summary()
    return model_vgg16_tl


def resnet_model():
    resnet = keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        pooling=None,
        classes=2,
        classifier_activation='softmax',
        input_shape=(64, 64, 3)
    )

    model_resnet = keras.Sequential(
        [
            resnet,
            keras.layers.Flatten(),
            keras.layers.Dense(1500, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(150, activation='relu'),
            keras.layers.Dense(4, "softmax")
        ]
    )
    model_resnet.summary()
    return model_resnet

def main(dataset_path):
    images, labels, label_encoder = load_dataset(dataset_path)
    class_labels = label_encoder.classes_

    main = preprocess_input(np.array(images))
    X_train, X_test, Y_train, Y_test = train_test_split(main, labels, test_size=0.15, random_state=42)

    Y_train = to_categorical(Y_train, num_classes=len(class_labels))
    Y_test = to_categorical(Y_test, num_classes=len(class_labels))

    print(cyel, X_train.shape, X_test.shape, Y_train.shape, Y_test.shape, cres)

    model = vgg_model()

    num = 8
    plt.figure()
    print(Y_train[num])
    plt.imshow(X_train[num])

    model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.fit(X_train, Y_train, epochs = 5)

    model_resnet = resnet_model()    
    model_resnet.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model_resnet.fit(X_train, Y_train, epochs = 5)


if __name__ == '__main__':
    print(c_blue, "Hell o ...", cres)
    if len(sys.argv) != 2:
        print("Usage: python comparaison.py dirname")
        exit(1)

    dataset_path = sys.argv[1]
    main(dataset_path)

