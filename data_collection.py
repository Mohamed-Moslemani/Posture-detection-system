import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import PIL
from PIL import Image
import os 
import requests
import keras 
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sklearn
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import cv2


def collect_data(label, num_images):
    cap = cv2.VideoCapture(0)
    os.makedirs(f'data/{label}', exist_ok=True)
    count = len(os.listdir("data/" + label))

    while count < num_images+count:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Collecting Data', frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        cv2.imwrite(f'data/{label}/{count}.jpg', frame)
        count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    label = input("Enter label (good/bad): ")
    num_images = int(input("Enter number of images to capture: "))
    collect_data(label, num_images)
