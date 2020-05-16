import glob
import numpy as np
import cv2
#function to generate a numpy list of images
def generate_image_list(folder,target_size):
    file_list = glob.glob("E:/machine learning/comparison project/images/" + '*.jpg')
    file_list.extend(glob.glob("E:/machine learning/comparison project/images/" + '*.jpeg'))
    file_list.extend(glob.glob("E:/machine learning/comparison project/images/" + '*.png'))
    saved_files=[]


    print("found ", str(len(file_list)),"files in folder",str(folder))
    images=[]
    print("generating resized images")
    list_length=len(file_list)
    for i in range(list_length):
        print(str(i),str(i/list_length)+"% done")
        try:
            read = cv2.imread(file_list[i])
            shape_length = len(np.shape(read))
            if shape_length < 3:
                img_arr = cv2.cvtColor(read, cv2.COLOR_GRAY2BGR)
            elif shape_length > 3:
                img_arr = read[:, :, 0:3]
            else:
                img_arr = read
            resize=cv2.resize(img_arr,target_size)
            images+=[resize]
            saved_files+=[file_list[i]]
        except:
            print("image read error trying another image")
            continue
    return images,saved_files

images,names=generate_image_list('E:/duplicate filtered out/',(112,112))
img=np.uint8(np.array(images))
names=np.array(names)
#img=np.load("photos.npy")
np.save("C:/numy/A4_112.npy",img)
np.save("C:/numy/A4_112_names.npy",names)
