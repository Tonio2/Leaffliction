import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow
import keras
from keras import ops
from train import preprocess_img


def render(img1, img2, state):
    text = "Class predicted : " + state
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    combined_img = np.concatenate((img1_rgb, img2_rgb), axis=1)

    plt.figure(figsize=(10, 5))
    plt.imshow(combined_img)

    plt.axis('off')
    plt.figtext(0.5, 0.01, text, ha='center', fontsize=12, color='black', weight='bold')
    plt.tight_layout()
    plt.show()


def predict(img, fruit):
    path = "results/" + fruit + ".keras"
    print(path)
    if os.path.isfile(path) is False:
        print(f"Error: No model found for the fruit {fruit}")
        exit(1)
    print("Model reconstructed !")
    reconstructed_model = keras.models.load_model(path)

    # # Y_pred_prob = reconstructed_model.predict(X_test)
    # # Y_pred = np.argmax(Y_pred_prob, axis=1)
    # # Y_test_classes = np.argmax(Y_test, axis=1)

    # np.testing.assert_allclose(model.predict(img), reconstructed_model.predict(img))

    predictions = reconstructed_model.predict(img_array)
    return "Healthy"



# Need to protect split
def get_fruit(src):
    directory = os.path.dirname(src)
    directory_name = os.path.basename(directory)
    fruit = directory_name.split('_')[0]
    return fruit


def main(src):
    # check if is file
    if os.path.isfile(src) is False:
        print("Not a file")
        exit(1)

    try:
        img1 = cv2.imread(src)
        if img1 is None:
            raise FileNotFoundError("Image not found or cannot be read.")

        fruit = get_fruit(src)

        # Apply transformation on img2
        # img2 = preprocess_img(src, img_size=256)

        # should send img2
        state = predict(img1, fruit)
        # render(img1, img1, state)

    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python predict.py 'path_image'")
        exit(1)
    main(sys.argv[1])
