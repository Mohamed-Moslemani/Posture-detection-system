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

print(len(os.listdir('data/good')))