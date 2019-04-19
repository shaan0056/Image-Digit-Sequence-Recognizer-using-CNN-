
from unpacker import DigitStructWrapper
import pandas as pd
import numpy as np
import cv2
import os

TRAIN_PATH = "../Data/train/"
TEST_PATH = "../Data/test/"
EXTRA_PATH = "../Data/extra/"
IMAGE_SIZE = (32,32)

PROCESSED_TRAIN_PATH = "../Data/processed/train/"
PROCESSED_VALID_PATH = "../Data/processed/valid/"
PROCESSED_TEST_PATH  = "../Data/processed/test/"

def load_digit_struct(path):

    return DigitStructWrapper(path).unpack_all()


def convert_to_dataframe(digit_struct,path):

    return pd.DataFrame([{
        'file': path + image['filename'],
        'label': box['label'],
        'width': box['width'],
        'height': box['height'],
        'top': box['top'],
        'left': box['left']} for image in digit_struct for box in image['boxes']])


def perform_aggregation(dataframe):

    dataframe['bottom'] = dataframe['top']  + dataframe['height']
    dataframe['right']  = dataframe['left'] + dataframe['width']

    dataframe.drop(['height' , 'width'],axis=1,inplace=True)

    dataframe =  dataframe.groupby('file', as_index=False).agg(
        {'top':'min','left':'min','bottom':'max','right':'max'
         ,'label': {'labels':lambda x: list(x),'digit_count':'count'}}
    )

    dataframe.columns = [x[0] if index != 6 else x[1] for index, x  in enumerate(dataframe.columns.values)]

    return dataframe

def draw_image(img,coordinates,delay=0,image_name=None):

    cv2.rectangle(img,coordinates[0],coordinates[1], (255,0,0), 2)
    cv2.imshow("image" if image_name is None else image_name,img)
    cv2.waitKey(delay)


def draw_random_image(dataframe,count=1):

    random_num = np.random.randint(dataframe.shape[0],size=count)
    for i in random_num:

        draw_image(cv2.imread(dataframe['file'].ix[i]),
                   [(int(dataframe['left'].ix[i]),int(dataframe['top'].ix[i])),
                    (int(dataframe['right'].ix[i]),int(dataframe['bottom'].ix[i]))])


def expand_image(dataframe):

    dataframe['width_expand']  = (0.3 * (dataframe['right'] - dataframe['left']))  / 2.
    dataframe['height_expand'] = (0.3 * (dataframe['bottom']  - dataframe['top'])) / 2.

    dataframe['left']  -= dataframe['width_expand'].astype('int')
    dataframe['right'] += dataframe['width_expand'].astype('int')

    dataframe['top']    -= dataframe['height_expand'].astype('int')
    dataframe['bottom'] += dataframe['height_expand'].astype('int')

    dataframe.drop(['width_expand','height_expand'],axis=1,inplace=True)

    return dataframe


def get_image_size(dataframe):

    file_names = dataframe['file'].tolist()

    image_size = [(cv2.imread(name).shape[:2]) for name in file_names]

    image_x_size = [x for (x,y) in image_size]
    image_y_size = [y for (x, y) in image_size]

    dataframe['image_height'] = image_x_size
    dataframe['image_width']  = image_y_size

    return dataframe


def correct_boundaries(dataframe):

    dataframe['top'].loc[dataframe['top'] < 0] = 0
    dataframe['top'].loc[dataframe['left'] < 0] = 0

    dataframe['bottom'].loc[dataframe['bottom'] > dataframe['image_height']] = dataframe['image_height']
    dataframe['right'].loc[dataframe['right']  > dataframe['image_width']]  = dataframe['image_width']

    return dataframe


def crop_image_and_save(dataframe,path,new_size):

    metadata = []

    for count, (index, rows) in enumerate(dataframe.iterrows()):
        try:
                image = cv2.imread(rows['file'])
                if rows['label'] == 10:
                    rows['label'] = 0.0
                directory = path + str(rows['label']) + "/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                file_name = directory + str(count) + ".png"
                if rows['left'] < 0:
                    rows['left'] = 0
                if rows['top'] < 0:
                    rows['top'] = 0
                cropped_image = image[int(rows['top']):int(rows['top']) + int(rows['height'])
                , int(rows['left']):int(rows['left']) + int(rows['width'])]

                cropped_image = cv2.resize(cropped_image, new_size)

                cv2.imwrite(file_name, cropped_image)

                metadata.append({'file': file_name, 'label': np.array(rows['label'])})
        except:
            print("Error at row --> " + str(rows))


    pd.DataFrame(metadata).to_csv(path+'metadata.csv')


if __name__ == "__main__":

    # train_digit_struct  = load_digit_struct(TRAIN_PATH + "digitStruct.mat")
    # test_digit_struct   = load_digit_struct(TEST_PATH  + "digitStruct.mat")
    # valid_digit_struct  = load_digit_struct(EXTRA_PATH + "digitStruct.mat")
    #
    #
    # convert_to_dataframe(train_digit_struct, TRAIN_PATH).to_csv(TRAIN_PATH + "train.csv")
    # convert_to_dataframe(test_digit_struct, TEST_PATH ).to_csv(TEST_PATH  + "test.csv")
    # convert_to_dataframe(valid_digit_struct, EXTRA_PATH).to_csv(EXTRA_PATH + "extra.csv")



    train_df = pd.read_csv(TRAIN_PATH + "train.csv")
    extra_df = pd.read_csv(EXTRA_PATH + "extra.csv")
    train = pd.concat([train_df,extra_df])

    train_df = train.sample(frac=0.99)
    valid_df = train.loc[~train.index.isin(train_df.index)]

    test_df = pd.read_csv(TEST_PATH + "test.csv")

    # train_df = perform_aggregation(train_df)
    # test_df  = perform_aggregation(test_df)

   # draw_random_image(train_df)
    #draw_random_image(test_df)

    #increase the gox by 30%

    # train_df = expand_image(train_df)
    # test_df = expand_image(test_df)

    #draw_random_image(train_df)
    #draw_random_image(test_df)


    #append image size

    # train_df = get_image_size(train_df)
    # test_df = get_image_size(test_df)

    #correct the expanded bounding box

    # train_df = correct_boundaries(train_df)
    # test_df  = correct_boundaries(test_df)

    #crop the images to 64 X 64

    crop_image_and_save(train_df,PROCESSED_TRAIN_PATH,IMAGE_SIZE)
    crop_image_and_save(valid_df, PROCESSED_VALID_PATH, IMAGE_SIZE)
    crop_image_and_save(test_df, PROCESSED_TEST_PATH, IMAGE_SIZE)























