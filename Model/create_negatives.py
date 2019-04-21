
import cv2
import numpy as np
import os, sys

image_path = "../Data/Negatives/"
dest_path_train = "../Data/Final/train/10.0/"

dest_path_test = "../Data/Final/test/10.0/"
dest_path_valid = "../Data/Final/valid/10.0/"

train_count = 45000
test_count  = 2000
valid_count = 500
window = (32,32)
crop_out = (200,200)

if not os.path.exists(dest_path_train):
    os.mkdir(dest_path_train)
if not os.path.exists(dest_path_train):
    os.mkdir(dest_path_test)
if not os.path.exists(dest_path_train):
    os.mkdir(dest_path_valid)

negative_images = os.listdir(image_path)
image_count = 0

for images in negative_images:

    img = cv2.imread(image_path+images)
    for row in range(0, img.shape[0] - crop_out[0]):
        for col in range(0, img.shape[1] - crop_out[1]):

            crop = img[row:row + crop_out[0], col:col + crop_out[1]]
            crop = cv2.resize(crop,window)
            image_count +=1
            if image_count <= train_count:
                cv2.imwrite(dest_path_train + "{}.png".format(image_count),crop)
            if image_count > train_count and image_count <= train_count + test_count:
                cv2.imwrite(dest_path_test + "{}.png".format(image_count),crop)
            if image_count > train_count + test_count and \
                    image_count <= train_count + test_count + valid_count:
                cv2.imwrite(dest_path_valid + "{}.png".format(image_count),crop)













