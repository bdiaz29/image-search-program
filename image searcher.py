from tkinter import *
import PIL
from PIL import Image, ImageGrab, ImageTk
import numpy as np
from tkinter import filedialog
import os.path
from os import path
import xlwt
from tkinter import messagebox
import cv2
import os.path
import glob
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K
import imutils
import xlwt
import pandas as pd
import xlsxwriter
from tkinter.ttk import Progressbar
import time

#empty arrays to fill in later
images = []
file_names = []
model = []

match_filename = []
match_value = []
match_image = []
list_length = 0
# set up tkinter winsow
window = Tk()
window.title("image search application")
window.geometry('1300x900')

#blank picture to start as default place holders
blank = np.zeros((112, 112, 3))
blank_img = Image.fromarray(np.uint8(np.array(blank)))
tk_blank = ImageTk.PhotoImage(blank_img)

blank2 = np.zeros((480, 640, 3))
blank_img2 = Image.fromarray(np.uint8(np.array(blank2)))
tk_blank2 = ImageTk.PhotoImage(blank_img2)

#function to create an npz file
def create_npz():
    #the size of the pictures
    target_size = (112, 112)
    source = filedialog.askdirectory(title="select source image folder") + "/"
    files = [('npz Files', '*.npz')]
    destination = filedialog.asksaveasfile(filetypes=files, defaultextension=files)
    # get all the file names
    file_list = glob.glob(source + '*.jpg')
    file_list.extend(glob.glob(source + '*.jpeg'))
    file_list.extend(glob.glob(source + '*.png'))
    # set up the image and names array
    images = []
    file_names = []
    list_length = len(file_list)
    #set up temporary progress bar and message
    temp_lbl = Label(console_frame, text="creating npz file")
    temp_lbl.grid(column=0, row=8, padx=5, pady=5)
    progress = Progressbar(console_frame, orient=HORIZONTAL,
                           length=600, mode='determinate')
    progress.grid(column=0, row=9, padx=5, pady=5)

    for i in range(list_length):
        print(str(i), str(i / list_length) + "% done")
        #update progress bar
        done = int((i / list_length) * 60)
        progress['value'] = done
        window.update()
        try:
            read = cv2.imread(file_list[i])
            shape_length = len(np.shape(read))
            #if image is grayscale
            if shape_length < 3:
                img_arr = cv2.cvtColor(read, cv2.COLOR_GRAY2BGR)
            #if image is a png
            elif shape_length > 3:
                img_arr = read[:, :, 0:3]
            #if regular jpg image
            else:
                img_arr = read
            resize = cv2.resize(img_arr, target_size)
            images += [resize]
            file_names += [file_list[i]]
        except:
            print("image read error trying another image")
            continue
    #remove tempory windows
    progress.destroy()
    temp_lbl.configure(text="converting images to numpy arrays")
    image = np.array(images)
    temp_lbl.configure(text="converting file paths to numpy arrays")
    file_names = np.array(file_names)
    temp_lbl.configure(text="saving arrays")
    np.savez(destination.name, images=images, file_names=file_names)
    temp_lbl.destroy()


# to load npz files
def load_npz():
    global images, file_names
    source = filedialog.askopenfilename(initialdir="/", title="Select file",
                                        filetypes=(("npz files", "*.npz"), ("all files", "*.*")))

    temp_lbl = Label(console_frame, text="loading npz file ")
    temp_lbl.grid(column=0, row=7, padx=5, pady=5)

    data = np.load(source)
    images = data['images']
    file_names = data['file_names']
    temp_lbl.destroy()


def load_h5():
    global model
    source = filedialog.askopenfilename(initialdir="E:/machine learning/saved models/", title="Select file",
                                        filetypes=(("h5 files", "*.h5"), ("all files", "*.*")))
    model = tf.keras.models.load_model(source, custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU(alpha=0.3)},
                                       compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )


def clear_images():
    global pics, pics_labels, preview_pic
    for i in range(40):
        pics[i].configure(image=tk_blank)
        pics[i].image = tk_blank
        pics_labels[i].configure(text=str("empty"))
    preview_pic.config(image=tk_blank2)
    preview_pic.image = tk_blank2


def load_image():
    global images, file_names, model
    global match_filename, match_value, match_image
    global pics, pics_labels
    # clear pic labels
    clear_images()
    match_value = []
    match_image = []
    match_filename = []

    source = filedialog.askopenfilename(initialdir="/", title="Select file",
                                        filetypes=(("jpg files", "*.jpg"), ("all files", "*.*")))
    threshold = float(set_txt.get())
    image = cv2.imread(source)
    image = cv2.resize(image, (112, 112))
    # run image batch inference
    start=time.time()
    image_search_npy_batch(images, file_names, image, model, threshold=threshold, batch_size=512)
    #image_search_npy_batch(images, file_names, image, model, threshold=threshold, batch_size=64)
    end=time.time()
    print("total runtime",str(end-start))


def image_search_npy_batch(img_list, image_list_names, img, model, threshold=.5, batch_size=64):
    global match_filename, match_value, console_frame, match_image
    global window,console_frame
    length = len(img_list)
    count = 0
    prepared_batch = []
    image_batch = []

    steps = int(length / batch_size)
    remainder = length % batch_size
    print(str(steps), "steps remainder of ", str(remainder))
    transfer = 0

    temp_lbl = Label(console_frame, text="searching for image")
    temp_lbl.grid(column=0, row=7, padx=5, pady=5)
    progress = Progressbar(console_frame, orient=HORIZONTAL,
                           length=600, mode='determinate')
    progress.grid(column=0, row=8, padx=5, pady=5)

    for k in range(steps):
        done = int((k / steps) * 100)
        progress['value'] = done
        console_frame.update()
        window.update()
        print(str(k))
        names = image_list_names[k * batch_size:(k * batch_size) + batch_size]
        first = img_list[k * batch_size:(k * batch_size) + batch_size]
        second = batch_concat(img, first)
        third = second / 255
        pred = model.predict(third)
        for j in range(batch_size):
            if pred[j] > threshold:
                print("match at", str(names[j]))
                match_filename += [names[j]]
                temp = pred[j]
                value = temp[0]
                match_value += [value]
                match_image += [first[j]]
                test = pred[j]
                progress['value'] = done
                console_frame.update()
        update_images()

    print("finishing remainder")

    names = image_list_names[(k * batch_size) + batch_size:len(img_list)]
    first = img_list[(k * batch_size) + batch_size:len(img_list)]
    second = batch_concat(img, first)
    third = second / 255
    pred = model.predict(third)
    for j in range(len(pred)):
        if pred[j] > threshold:
            print("match at", str(names[j]))
            match_filename += [names[j]]
            match_value += [pred[j]]
            match_image += [first[j]]
            progress['value'] = done
            console_frame.update_idletasks()
    update_images()

    temp_lbl.destroy()
    progress.destroy()


def update_images():
    global match_filename, match_value, match_image
    global pics, pics_labels,window
    length = len(match_value)
    # temp=match_image
    if length <= 0:
        return
    temp = np.array(match_value)
    # arrange from highest to lowest
    sorted_index = np.argsort(-temp)
    # truncate to 40
    length = min(length, 40)
    for i in range(length):
        # set the image
        # first convert into rgb
        index = sorted_index[i]
        image_rgb = cv2.cvtColor(match_image[index], cv2.COLOR_BGR2RGB)
        # then convert into a PIL
        image_PIL = Image.fromarray(image_rgb)
        # then convert into tk image
        image_tk = ImageTk.PhotoImage(image_PIL)
        pics[i].configure(image=image_tk)
        pics[i].image = image_tk
        # then change label
        pics_labels[i].configure(text=str(match_value[index]))
        window.update()


def batch_concat(img, img_list):
    return_arr = []
    length = len(img_list)
    for c in range(length):
        temp1 = img_list[c]
        temp = cv2.vconcat([img, temp1])
        return_arr += [temp]

    return_arr = np.array(return_arr)
    return return_arr


def data():
    global pic_list_in
    global test_txt, pics

    for i in range(50):
        Label(pic_list_in, text=i).grid(row=i, column=0)
        Label(pic_list_in, text="my text" + str(i)).grid(row=i, column=1)
        Label(pic_list_in, text="..........").grid(row=i, column=2)


def picture_clicked(event):
    global preview_pic,match_filename
    pic_pos = int(event.widget.name)
    if pic_pos >= len(images):
        print("non existent image")
        return
    global match_filename, match_value, match_image
    global pics, pics_labels
    length = len(match_value)
    # temp=match_image
    if length <= 0:
        return
    temp = np.array(match_value)
    # arrange from highest to lowest
    sorted_index = np.argsort(-temp)
    # set the image
    index = sorted_index[pic_pos]
    path=match_filename[index]
    #load file
    image=cv2.imread(path)
    #resize
    image_resize=cv2.resize(image,(480,640))
    #convert to rgb
    image_rgb = cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB)
    # then convert into a PIL
    image_PIL = Image.fromarray(image_rgb)
    # then convert into tk image
    image_tk = ImageTk.PhotoImage(image_PIL)
    preview_pic.configure(image=image_tk)
    preview_pic.image = image_tk


def canvas_function(event):
    global pic_list_canvas
    pic_list_canvas.configure(scrollregion=pic_list_canvas.bbox("all"), width=600, height=600)


pic_title_lbl = Label(master=window, text='Results')
pic_title_lbl.grid(column=0, row=0)

console_frame = Frame(master=window)
console_frame.grid(column=1, row=1, sticky="NW")

pic_list_frame = Frame(master=window, height=700, width=700)
pic_list_frame.grid(column=0, row=1, padx=5, pady=5)

pic_list_canvas = Canvas(pic_list_frame)

pic_list_in = Frame(pic_list_canvas)

pic_scrollbar = Scrollbar(pic_list_frame, orient='vertical', command=pic_list_canvas.yview)
pic_list_canvas.configure(yscrollcommand=pic_scrollbar.set)

pic_scrollbar.pack(side='right', fill='y')
pic_list_canvas.pack(side='left')
pic_list_canvas.create_window((0, 0), window=pic_list_in, anchor='nw')
pic_list_in.bind('<Configure>', canvas_function)

# set up 40 pic and label planes
pics = []
pics_labels = []
count = 0
for r in range(10):
    for c in range(4):
        pics += [Label(master=pic_list_in, image=tk_blank, name=str(count))]
        pics[count].bind("<Button-1>", picture_clicked)
        pics[count].grid(column=c, row=r * 2)
        pics[count].name = str(count)
        pics_labels += [Label(master=pic_list_in, text="empty")]
        pics_labels[count].grid(column=c, row=(r * 2) + 1)
        count = count + 1
load_npz_btn = Button(master=console_frame, text="load npz", command=load_npz)
load_npz_btn.grid(column=0, row=0, padx=5, pady=5, sticky="W")

create_npz_btn = Button(master=console_frame, text="create npz", command=create_npz)
create_npz_btn.grid(column=0, row=1, padx=5, pady=5, sticky="W")

load_h5_btn = Button(master=console_frame, text="load h5", command=load_h5)
load_h5_btn.grid(column=0, row=2, padx=5, pady=5, sticky="W")

set_lbl = Label(master=console_frame, text="set threshold:")
set_lbl.grid(column=0, row=3, padx=5, pady=5, sticky="W")

set_txt = Entry(master=console_frame)
set_txt.grid(column=0, row=4, padx=5, pady=5, sticky="W")
set_txt.insert(END, ".9")

load_image_btn = Button(master=console_frame, text="load image", command=load_image)
load_image_btn.grid(column=0, row=5, padx=5, pady=5)

preview_lbl = Label(master=console_frame, text="image preview")
preview_lbl.grid(column=0, row=6, padx=5, pady=5)

preview_pic = Label(master=console_frame, image=tk_blank2)
preview_pic.grid(column=0, row=7, padx=5, pady=5)

window.mainloop()
