import sys
import os
import tensorflow
import keras
import numpy as np
import matplotlib.pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transfo2 import gaussian_blur, mask_objects
import Augmentation
import cv2
from plantcv import plantcv as pcv


def count(set):
    counts = {
        'Apple_healthy': 0,
        'Apple_scab': 0,
        'Apple_Black_rot': 0,
        'Apple_rust': 0,        
    }
    for img, label in set:
        counts[label] += 1
        
    print(counts)

def call_augmentation(local_dir, name_dir):


    v = os.path.join("./dataset/", name_dir + "/")
    try:
        print(f"v in try : {v}")
        os.mkdir(v)
    except FileExistsError:
        print("Directory 'dataset' already exists.")
    except PermissionError:
        print("Permission denied: Unable to create dataset.")
        exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)


    dirs = [d for d in local_dir if Augmentation.is_dir(v, d)]
    print(dirs)
    # dir_size = [len(os.listdir(os.path.join(new_dir, d))) for d in dirs]
    # print(dir_size)
    # max_dir_size = max(dir_size)

    # for d in dirs:
    #     idx = 0
    #     img_list = os.listdir(os.path.join(new_dir, d))
    #     dir_size = len(img_list)
    #     while dir_size + 6 * idx <= max_dir_size:
    #         img = os.path.join(new_dir, d, img_list[idx])
    #         print(img)
    #         modifs = Augmentation.augmentation(img)
    #         for key, value in modifs.items():
    #             save_path = img.split(".JPG")[0] + key
    #             cv2.imwrite(save_path, value)
    #         idx += 1


def dataset(dirname):
    try:
        os.mkdir("dataset")
    except FileExistsError:
        print("Directory 'dataset' already exists.")
    except PermissionError:
        print("Permission denied: Unable to create dataset.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
    dirs = [d for d in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, d))]
    array = []
    for d in dirs:
        for img in os.listdir(os.path.join(dirname, d)):
            array.append((os.path.join(dirname, d, img), d))
    
    count(array)
    
    list_img = []
    # Transformation
    for img in array:
        img_path, label = img
        img = cv2.imread(img_path)
        img = pcv.rgb2gray(img)
        cv2.imwrite(img_path, img)
        list_img.append((img, label))
        # print(f"Transformed {img_path}")
    
    # Normalize
    images = np.array([i[0] for i in list_img], dtype="float32") / 255.0
    labels = [i[1] for i in list_img]
    
    # Labelize
    le = LabelEncoder()
    le.fit(labels)
    labels_encoder = le.fit_transform(labels)
    print(labels_encoder)

    
    # Split
    # dirs = [d for d in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, d))]
    # array = []
    # for d in dirs:
    #     for img in os.listdir(os.path.join(dirname, d)):
    #         array.append((os.path.join(dirname, d, img), d))
            
    X_training, X_test, Y_train, Y_test = train_test_split(images, labels_encoder, train_size=0.85, random_state=42)
    # X_train, X_validation = train_test_split(X_training, test_size=0.15, random_state=42)
    
    # Define model
    learning_rate_decay = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

    model = keras.models.Sequential([
        keras.Input(shape=(256,256, 1)),

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
    model.fit(x=X_training, y=Y_train, epochs=50, callbacks=[learning_rate_decay, early_stopping], validation_split=0.15)
    
    # Evaluate model


# def encoder():
# def classfier():

def main(dirname):
    dataset(dirname)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python train.py dirname")
        exit(1)
    main(sys.argv[1])
