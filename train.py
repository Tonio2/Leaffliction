import sys
import os
from tensorflow import keras
import numpy
import matplotlib.pyplot
import sklearn as sk

def dataset(dirname):
	X_train, X_validation = sk.model_selection.train_test_split(dirname, test_size=0.15, train_size=None, random_state=42, shuffle=True, stratify=None)

def main(dirname):
	dataset(dirname)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python train.py dirname")
        exit(1)
    main(sys.argv[1])
