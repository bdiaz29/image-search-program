import xmltodict
import os
import glob
import xmltodict
import xlwt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import keras
import numpy as np
import random
from random import randint
import PIL
from PIL import Image
import cv2
import xlrd
import pandas as pd
import math
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPool2D
from keras.models import load_model
from datetime import datetime
import time
from tensorflow.keras.models import Model, Sequential, load_model
import cv2
import imutils

count2 = 0
datagen = ImageDataGenerator()
# xlfilepath = "C:/Users/Bruno/PycharmProjects/beta/Bdetect/breastdata.xls"
xlfilepath = "E:/machine learning/over.xls"
df = pd.read_excel(xlfilepath)
df_list = df.to_numpy()
file_list = np.delete(df_list, 0, 0)
np.random.shuffle(file_list)
shape = np.shape(file_list)
rows = shape[0]
num_list = [None] * rows

for i in range(rows):
    num_list = i
count = 0
limit = int(rows / 64)



def custom_loss(y_true, y_pred):
    mse = tf.losses.mean_squared_error(y_true, y_pred)
    iou = calculate_iou(y_true, y_pred)
    return mse + (1 - iou)


def iou_metric(y_true, y_pred):
    return calculate_iou(y_true, y_pred)


def biased(y_true, y_pred):
    return K.mean(y_pred)


def other(y_true, y_pred):
    return K.mean(y_true)


def biased_over_other(y_true, y_pred):
    A = K.mean(y_pred)
    B = K.mean(y_true)
    C = A / B
    return C


def generate_image_list(folder, target_size):
    file_list = glob.glob(folder + '*.jpg')
    file_list.extend(glob.glob(folder + '*.jpeg'))
    file_list.extend(glob.glob(folder + '*.png'))
    print("found ", str(len(file_list)), "files in folder", str(folder))
    images = []
    print("generating resized images")
    list_length = len(file_list)
    for i in range(list_length):
        print(str(i))
        try:
            read = cv2.imread(file_list[i])
            shape_length = len(np.shape(read))
            if shape_length < 3:
                img_arr = cv2.cvtColor(read, cv2.COLOR_GRAY2BGR)
            elif shape_length > 3:
                img_arr = read[:, :, 0:3]
            else:
                img_arr = read
            resize = cv2.resize(img_arr, target_size)
            images += [resize]
        except:
            print("image read error trying another image")
            continue
    return images

#custom data generator taking in a numpy array
def generate(images, batch_size=32, target_size=(112, 112)):
    while True:  # Select files (paths/indices) for the batch
        batch_input = []
        batch_output = []
        length_list = len(images)

        # Read in each input, perform preprocessing and get labels
        for index in range(batch_size):
            # randomly select an image
            random_num_img1 = randint(0, len(images) - 1)
            temp = images[random_num_img1]
            s = np.shape(temp)
            h = s[0]
            w = s[1]
            random_hor = randint(0, 8)
            random_ver = randint(0, 8)
            random_hor2 = randint(0, 8)
            random_ver2 = randint(0, 8)
            #randomly augment image with zoom and shift 
            img1 = temp[random_ver:h - random_ver2, random_hor:w - random_hor2]
            img1 = cv2.resize(img1, (112, 112))
            # put in a random contrast
            random_contrast = random.uniform(.75, 1.25)
            img1 = img1 * random_contrast
            img1 = np.uint8(img1)
            # rotate the image
            angle = random.randint(-4, 4)
            img1 = imutils.rotate(img1, angle)
            # 50% chance to use the same image
            random_num_0 = randint(0, 99)
            if random_num_0 <= 50:
                out = 1
                # 50% chance to shift augment the same image
                random_num_1 = randint(0, 99)
                if random_num_1 < 75:
                    #randomly create new image from the same
                    random_hor = randint(0, 8)
                    random_ver = randint(0, 8)
                    random_hor2 = randint(0, 8)
                    random_ver2 = randint(0, 8)
                    new_img = temp[random_ver:h - random_ver2, random_hor:w - random_hor2]
                    img2 = cv2.resize(new_img, (112, 112))
                    # put in a random contrast
                    random_contrast = random.uniform(.75, 1.25)
                    img2 = img2 * random_contrast
                    img2 = np.uint8(img2)
                    # rotate the image
                    angle = random.randint(-4, 4)
                    img2 = imutils.rotate(img2, angle)
                    out = 1
                else:
                    img2 = np.array(img1)
                    out = 1
            else:
                # choose another random image
                out = 0
                while True:
                    random_num_img2 = randint(0, len(images) - 1)
                    if random_num_img1 != random_num_img2:
                        # print("not duplicate",str(random_num_img1),str(random_num_img2))
                        break
                    else:
                        print(" duplicate", str(random_num_img1), str(random_num_img2))

                #img2 = images[random_num_img2]

                temp = images[random_num_img2]
                s = np.shape(temp)
                h = s[0]
                w = s[1]
                random_hor = randint(0, 8)
                random_ver = randint(0, 8)
                random_hor2 = randint(0, 8)
                random_ver2 = randint(0, 8)
                img2 = temp[random_ver:h - random_ver2, random_hor:w - random_hor2]
                img2 = cv2.resize(img2, (112, 112))
                # put in a random contrast
                random_contrast = random.uniform(.75, 1.25)
                img2 = img2 * random_contrast
                img2 = np.uint8(img2)
                # rotate the image
                angle = random.randint(-4, 4)
                img2 = imutils.rotate(img2, angle)
                out = 0
            # concatonate the images
            inp = cv2.vconcat([img1, img2]) / 255
            batch_input += [inp]
            batch_output += [out]  #
            # Return a tuple of (input, output) to feed the network
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)
        yield batch_x, batch_y


alpha = 0.3
input_shape = (224, 112, 3)
acti = tf.keras.layers.LeakyReLU(alpha=0.3)
model_layers = [
    # adding guassian noise
    tf.keras.layers.GaussianNoise(stddev=.1, input_shape=input_shape),
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=alpha),
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=alpha),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=alpha),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=alpha),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=alpha),
    tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=alpha),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=alpha),
    tf.keras.layers.Conv2D(1024, kernel_size=(3, 3), strides=1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=alpha),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=alpha),
    tf.keras.layers.Dense(256),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=alpha),
    tf.keras.layers.Dense(128),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=alpha),
    tf.keras.layers.Dense(128),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=alpha),
    tf.keras.layers.Dense(1, activation='sigmoid')
]
model = Sequential(model_layers)

batch_size = 64
data_size = 500000
steps = int(data_size / batch_size)

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'mean_absolute_error', biased, other, biased_over_other]
)
#use this to load whatever image data you have prepared
img = np.load("C:/numy/27443_112_nodup.npy")
#custom data generator
datagen = generate(img, batch_size=batch_size)

model.fit_generator(datagen, steps_per_epoch=steps, epochs=3, verbose=1)
model.save("E:/machine learning/saved models/customSmaller4.h5")
