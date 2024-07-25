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
import shutil

def augment_data(class_dir, output_dir, batch_size=32, img_height=150, img_width=150):
    if not os.path.exists(class_dir):
        print(f"Input directory {class_dir} does not exist.")
        return
        
    class_name = os.path.basename(os.path.normpath(class_dir))

    temp_dir = 'temp_data'
    temp_class_dir = os.path.join(temp_dir, class_name)
    os.makedirs(temp_class_dir, exist_ok=True)

    for filename in os.listdir(class_dir):
        src = os.path.join(class_dir, filename)
        dst = os.path.join(temp_class_dir, filename)
        shutil.copy(src, dst)  

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    generator = datagen.flow_from_directory(
        temp_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=None,
        save_to_dir=output_dir,
        save_prefix='aug',
        save_format='jpeg'
    )

    for i in range(20):  
        next(generator)

    for filename in os.listdir(temp_class_dir):
        os.remove(os.path.join(temp_class_dir, filename))
    os.rmdir(temp_class_dir)
    os.rmdir(temp_dir)

if __name__ == "__main__":
    class_folder = 'data/bad'
    augment_data(class_folder, 'data/augmented', batch_size=32, img_height=150, img_width=150)