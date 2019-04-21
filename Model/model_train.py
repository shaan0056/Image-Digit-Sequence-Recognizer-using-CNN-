
import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

PROCESSED_TRAIN_PATH = "../Data/Final/train/"
PROCESSED_VALID_PATH = "../Data/Final/valid/"
PROCESSED_TEST_PATH  = "../Data/Final/test/"
EPOCHS = 10
BATCH_SIZE = 128
INPUT_SIZE = (32,32,3)

#load data

# train_metadata = pd.read_csv(PROCESSED_TRAIN_PATH+'metadata.csv')
# test_metadata  = pd.read_csv(PROCESSED_TEST_PATH+'metadata.csv')



if __name__ == "__main__":

    vgg_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=INPUT_SIZE)


    model = tf.keras.models.Sequential()

    model.add(vgg_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(11))
    model.add(tf.keras.layers.Activation('softmax'))

    print(model.summary())

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                    width_shift_range=0.2,
                                                                    height_shift_range=0.2,
                                                                    zoom_range=0.2,
                                                                    brightness_range=[0.5,1.5],
                                                                    rotation_range=30)

    test_datagen  = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    train_loader = train_datagen.flow_from_directory(PROCESSED_TRAIN_PATH,batch_size=BATCH_SIZE, target_size=INPUT_SIZE[:2])
    valid_loader = train_datagen.flow_from_directory(PROCESSED_VALID_PATH, batch_size=BATCH_SIZE,target_size=INPUT_SIZE[:2])

    Checkpoint = tf.keras.callbacks.ModelCheckpoint("myCNN.h5", monitor='val_acc', mode='max', verbose=0,
                                                 save_best_only=True, save_weights_only=False, period=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=0, mode='max')
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="log_dir", histogram_freq=10, batch_size=BATCH_SIZE, write_graph=True,
                                              write_grads=False, write_images=False, embeddings_freq=0,
                                              embeddings_layer_names=None, embeddings_metadata=None)

    history = model.fit_generator(
        train_loader,
        steps_per_epoch=train_loader.samples // BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=valid_loader,
        validation_steps=valid_loader.samples // BATCH_SIZE,
        callbacks=[Checkpoint,early_stopping])

    test_loader = test_datagen.flow_from_directory(PROCESSED_TEST_PATH, batch_size=BATCH_SIZE,target_size=INPUT_SIZE[:2])

    best_model = tf.keras.models.load_model("myCNN.h5")

    score = best_model.evaluate_generator(test_loader)

    print("Test Accuracy: {:.4f}".format(score[1]))

    with open('trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("accuracy.jpg")
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("loss.jpg")