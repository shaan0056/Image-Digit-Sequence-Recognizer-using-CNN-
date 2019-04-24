
from scipy.io import loadmat
import numpy as np
import cv2
import os

save_path = '../Data/cropped/'


def create_dir(*args):
    for directory in args:
        if not os.path.exists(directory):
            os.makedirs(directory)

def load_data(path):
    """ Helper function for loading a MAT-File"""
    data = loadmat(path)
    return data['X'], data['y']


def balanced_subsample(y, s):
    """Return a balanced subsample of the population"""
    sample = []
    # For every label in the dataset
    for label in np.unique(y):
        # Get the index of all images with a specific label
        images = np.where(y == label)[0]
        # Draw a random sample from the images
        random_sample = np.random.choice(images, size=s, replace=False)
        # Add the random sample to our subsample list
        sample += random_sample.tolist()
    return sample


def save_image(images,label,folder):
    for i, img in enumerate(images):

        path = save_path+folder+str(label[i])
        create_dir(path)
        cv2.imwrite(path+'/{}.png'.format(i),img)


if __name__ == '__main__':

    create_dir(save_path)

    X_train, y_train = load_data('../Data/train_32x32.mat')
    X_test, y_test = load_data('../Data/test_32x32.mat')
    #X_extra, y_extra = load_data('../Data/extra_32x32.mat')

    X_train, y_train = X_train.transpose((3,0,1,2)), y_train[:,0]
    X_test, y_test = X_test.transpose((3,0,1,2)), y_test[:,0]
    #X_extra, y_extra = X_extra.transpose((3,0,1,2)), y_extra[:,0]


    train_samples = balanced_subsample(y_train, 1000)
    # Pick 200 samples per class from the extra dataset
    #extra_samples = balanced_subsample(y_extra, 200)

    X_val, y_val = np.copy(X_train[train_samples]), np.copy(y_train[train_samples])

    # Remove the samples to avoid duplicates
    X_train = np.delete(X_train, train_samples, axis=0)
    y_train = np.delete(y_train, train_samples, axis=0)

    #X_val = np.concatenate([X_val, np.copy(X_extra[extra_samples])])
    #y_val = np.concatenate([y_val, np.copy(y_extra[extra_samples])])

    # Remove the samples to avoid duplicates
    #X_extra = np.delete(X_extra, extra_samples, axis=0)
    #y_extra = np.delete(y_extra, extra_samples, axis=0)

    #X_train = np.concatenate([X_train, X_extra])
    #y_train = np.concatenate([y_train, y_extra])
    X_test, y_test = X_test, y_test

    save_image(X_train,y_train,'train/')
    save_image(X_test,y_test,'test/')
    save_image(X_val, y_val, 'valid/')





