
import cv2
import numpy as np
import tensorflow as tf
import os

out_path = 'graded_images/'

#def non_maximum_suppression():



def image_pyramid(img,scale_up=0,scale_down=0,step=.1):

    pyramid = []
    height,width = img.shape[:2]
    ratio = 1.0

    for i in range(scale_up):

        ratio += step
        img_up = cv2.resize(img,dsize=(int(width * ratio), int(height * ratio)))
        pyramid.append((ratio,img_up))

    pyramid.append((1,img))

    ratio = 1.0
    for i in range(scale_down):

        ratio -= step
        img_down = cv2.resize(img,dsize=(int(width * ratio), int(height * ratio)))
        pyramid.append((ratio,img_down))

    return pyramid


def sliding_window(img,classifier,detector,window=(32,32)):

    bounding_box = {}

    for row in range(0,img.shape[0] - window[0],32):#32 works
        for col in range(0,img.shape[1] - window[1],32):

            cropped_img = img[row:row + window[0], col:col + window[1]]
            cropped_img = cv2.resize(cropped_img,(32,32))

            cropped_img = np.expand_dims(cropped_img, axis=0)

            detection = detector.predict(cropped_img)
            #print ("detect -> " ,detection[0][1])
            prediction = classifier.predict_proba(cropped_img)


            if max(prediction[0]) == 1. and detection[0][1] == 1.:
                label = np.argmax(prediction[0])
                others = [prediction[0][i] for i in range(len(prediction[0])) if i != label]
                if sum(others) == 0. and label != 10:
                    print(label, row, col)
                    bounding_box[label] = (row,col)

    return bounding_box


def create_dir(*args):
    for directory in args:
        if not os.path.exists(directory):
            os.makedirs(directory)



if __name__ == '__main__':


    create_dir(out_path)
    img = cv2.imread("../Data/input_images/1.jpg")
    copied_image = img.copy()
    (h,w) = copied_image.shape[:2]
    img = cv2.resize(img,(200,200))

    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # edged = cv2.Canny(blurred, 50, 200, 255)

    #img = cv2.GaussianBlur(img,(5,5),0)

    #img = img[80:112,180:192]

    #
    mser = cv2.MSER_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15, 15), 0)
    vis = img.copy()
    regions, _ = mser.detectRegions(gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(vis, hulls, 1, (0, 255, 0))
    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    mask = cv2.dilate(mask, np.ones((150, 150), np.uint8))
    for contour in hulls:
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

        text_only = cv2.bitwise_and(img, img, mask=mask)
    #
    # cv2.imshow('img', vis)
    # cv2.waitKey(0)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    # cv2.imshow('text', text_only)
    # cv2.waitKey(0)


    #
    # cv2.imshow('test', edged)
    # cv2.waitKey(0)

    #
    classifier = tf.keras.models.load_model("myCNN.h5")
    detector = tf.keras.models.load_model("myDetectorCNN.h5")
    #img = np.expand_dims(img, axis=0)
    #prediction = classifier.predict(img)
    #print(prediction)


    # print(prediction)

    bounding_box = sliding_window(text_only.copy(),classifier,detector)



    for label,(height,width) in bounding_box.items():



        cv2.rectangle(img,(width,height),(width + 32,height+32),(255,0,0),1)
        cv2.putText(img,str(label),(width-32 ,height+32),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    img = cv2.resize(img,(w,h))
    cv2.imwrite(out_path+"test.jpg",img)
    # cv2.imshow('test',img)
    # cv2.waitKey(0)
