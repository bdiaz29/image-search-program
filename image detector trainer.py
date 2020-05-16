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


def debug_func(img_arr, xmin, ymin, xmax, ymax):
    S = np.shape(img_arr)
    height = S[0]
    width = S[1]
    x_scale = width / 224
    y_scale = height / 224
    im = Image.fromarray(img_arr)
    im2 = im.resize((224, 224))
    im3 = np.array(im2)
    xmin = xmin * x_scale
    ymin = ymin * y_scale
    xmax = xmax * x_scale
    ymax = ymax * y_scale

    return im3, xmin, ymin, xmax, ymax


def image_generator(xlfilepath, batch_size=64):
    global file_list
    global count
    global count2
    global rows
    augment = False
    while True:  # Select files (paths/indices) for the batch
        if count >= (rows - (batch_size + 1)):
            count = randint(0, batch_size)

        batch_path_indexes = np.random.choice(a=num_list, size=batch_size, replace=False)
        batch_input = []
        batch_output = []

        # Read in each input, perform preprocessing and get labels
        for index in range(batch_size):
            # number = randint(0, 100)
            # if number < 10:
            #    augment = True
            # augment=False
            augment = False
            input, output = get_in_out(file_list[index + count, 0], file_list[index + count, 1],
                                       file_list[index + count, 2],
                                       file_list[index + count, 3], file_list[index + count, 4],
                                       [224, 224], augment)
            augment = False
            batch_input += [input]
            batch_output += [output]  #
            # Return a tuple of (input, output) to feed the network
        count = count + batch_size
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)
        yield batch_x, batch_y


# get image path augment image, get new points and preprocess the image
def get_in_out(img_path, xmin, ymin, xmax, ymax, target_size, augment):
    im = Image.open(img_path)
    # if augment not
    if augment == False:
        target_height = target_size[0]
        target_width = target_size[1]
        shape = np.shape(np.array(im))
        original_height = shape[0]
        original_width = shape[1]
        x_scale = target_height / original_height
        y_scale = target_width / original_width
        # determin new scale values
        xmin_scaled = xmin * x_scale
        ymin_scaled = ymin * y_scale
        xmax_scaled = xmax * x_scale
        ymax_scaled = ymax * y_scale
        # resize image
        im2 = im.resize((target_height, target_width))
        output_points = [xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled, 0]
        im2 = im.resize((224, 224))
        im3 = np.array(im2) / 255
        return im3, output_points
    img_arr = np.array(im)
    new = aug(img_arr, [xmin, ymin, xmax, ymax])
    new_img = new[0]
    new_points = new[1]
    xmin2 = new_points[0]
    ymin2 = new_points[1]
    xmax2 = new_points[2]
    ymax2 = new_points[3]

    xmin_scaled = xmin2
    ymin_scaled = ymin2
    xmax_scaled = xmax2
    ymax_scaled = ymax2

    if xmin_scaled <= 0:
        xmin_scaled = 0
    if ymin_scaled <= 0:
        ymin_scaled = 0
    if xmax_scaled > 1:
        xmax_scaled = .99
    if ymax_scaled > 1:
        ymax_scaled = .99
    new_img_processed = new_img / 255

    output_points = [xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled, 0]

    return new_img_processed, output_points


def aug(image_input, bounds):
    keep_within_bounds = True
    numb = randint(0, 99)
    if numb <= 50:
        keep_within_bounds = False

    img_arr = np.array(image_input)
    original_shape = np.shape(img_arr)
    original_height = original_shape[0]
    original_width = original_shape[1]
    original_height_midpoint = int(original_height / 2)
    original_width_midpoint = int(original_width / 2)
    original_x1 = int(bounds[0] * original_width)
    original_y1 = int(bounds[1] * original_height)
    original_x2 = int(bounds[2] * original_width)
    original_y2 = int(bounds[3] * original_height)
    original_xdiff = original_x2 - original_x1
    original_ydiff = original_y2 - original_y1
    # select points to take out of the original image
    if keep_within_bounds:
        left = random.randint(0, original_x1)
        top = random.randint(0, original_y1)
        # 50/50 chance to move next to from the right or the bottom
        chance_1 = random.randint(0, 99)
        if chance_1 < 50:
            right = random.randint(original_x2, original_width - 1)
            piece_length = abs(right - left)
            # trying to avoid image compressing or stretching more than a factor of 3
            small_limit = int(piece_length / 3) + top
            big_limit = int(piece_length * 3) + top
            lower = max(small_limit, original_y2)
            upper = min(big_limit, original_height - 1)
            bottom = random.randint(lower, upper)
        else:
            bottom = random.randint(original_y2, original_height - 1)
            piece_length = abs(top - bottom)
            # trying to avoid image compressing or stretching more than a factor of 3
            small_limit = int(piece_length / 3) + left
            big_limit = int(piece_length * 3) + left
            lower = max(small_limit, original_x2)
            upper = min(big_limit, original_width - 1)
            right = random.randint(lower, upper)
    else:
        left = random.randint(0, original_width_midpoint)
        top = random.randint(0, original_height_midpoint)
        right = random.randint(left + 1, original_width - 1)
        bottom = random.randint(top + 1, original_height - 1)

    # create new image from these
    new_img = img_arr[bottom:top, left:right]
    # adjust points

    x1 = (original_x1 - left) / 244
    y1 = (original_y1 - top) / 224
    x2 = (original_x2 - left) / 224
    y2 = (original_y2 - top) / 224

    if original_x1 <= left:
        x1 = 0
    if original_y1 <= top:
        y1 = 0
    if original_x2 <= left:
        x2 = 0
    if original_y2 <= top:
        y2 = 0

    if original_x1 >= right:
        x1 = .99
    if original_y1 >= bottom:
        y1 = .99
    if original_x2 >= right:
        x2 = .99
    if original_y2 >= bottom:
        y2 = .99

    new_points = [x1, y1, x2, y2]

    return new_img, new_points


#############################################################################################
# all below is the normal stuff
def calculate_iou(target_boxes, pred_boxes):
    xA = K.maximum(target_boxes[..., 0], pred_boxes[..., 0])
    yA = K.maximum(target_boxes[..., 1], pred_boxes[..., 1])
    xB = K.minimum(target_boxes[..., 2], pred_boxes[..., 2])
    yB = K.minimum(target_boxes[..., 3], pred_boxes[..., 3])
    interArea = K.maximum(0.0, xB - xA) * K.maximum(0.0, yB - yA)
    boxAArea = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
    boxBArea = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou


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


def random_read(file_list, target_size):
    while True:
        random_num = random.randint(0, len(file_list) - 1)
        try:
            read = cv2.imread(file_list[random_num])
            shape_length = len(np.shape(read))
            if shape_length < 3:
                img_arr = cv2.cvtColor(read, cv2.COLOR_GRAY2BGR)
            elif shape_length > 3:
                img_arr = read[:, :, 0:3]
            else:
                img_arr = read
            resize = cv2.resize(img_arr, target_size)
            break
        except:
            print("image read error trying another image")
    return resize


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
