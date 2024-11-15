import sys
import os
# import tensorflow
# import keras
import numpy
import matplotlib.pyplot
from sklearn.model_selection import train_test_split
import Augmentation
import cv2
from transfo2 import gaussian_blur, mask_objects

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
        
    # Transformation
    for img in array:
        img_path, label = img
        img = cv2.imread(img_path)
        img = gaussian_blur(img)
        img = mask_objects(img)
        cv2.imwrite(img_path, img)
        print(f"Transformed {img_path}")
    
    # Normalize
    
    # Labelize
    
    # Split
    # dirs = [d for d in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, d))]
    # array = []
    # for d in dirs:
    #     for img in os.listdir(os.path.join(dirname, d)):
    #         array.append((os.path.join(dirname, d, img), d))
            
    # X_training, X_test = train_test_split(array, train_size=0.85, random_state=42)
    # X_train, X_validation = train_test_split(X_training, test_size=0.15, random_state=42)
    
    # Define model
    
    # Train model
    
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
