import csv
import cv2
from keras.models import Sequential
from keras.layers import (
    BatchNormalization,
    Cropping2D,
    Dense,
    Dropout,
    Convolution2D,
    Flatten,
    Lambda,
    MaxPooling2D)
from keras.callbacks import ModelCheckpoint
import numpy as np
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

BATCH_SIZE = 32


def get_image(path):
    '''
    Given an image path, return the image
    '''
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class Frame(object):
    '''
    A single frame data which corresponds to a single line in the CSV file
    '''
    img_data = None

    def __init__(self, path, angle, flip=False):
        '''
        Converts an array of data into the object
        '''
        self.image_path = path
        self.angle = angle
        self.flip = flip

    @property
    def img(self):
        '''
        Returns the center image
        '''
        # if self.c_img_data is None:
        # self.c_img_data = get_image(self.c_img_path)
        # return self.c_img_data
        image = get_image(self.image_path)
        if self.flip:
            image = np.fliplr(image)

        return image

    def __repr__(self):
        '''
        String representation of a CVS data frame
        '''
        return '<Frame {image_path} {angle}'.format(
            image_path=self.image_path,
            angle=self.angle)


def get_frames():
    '''
    Returns the training data
    '''
    frames = []
    with open('./recorded_data/driving_log.csv') as csvfile:
        print('Opening file')
        reader = csv.reader(csvfile)
        print('Creating frames')
        for line in reader:
            if not line:
                continue

            c_path, l_path, r_path, steering, throttle, brake, speed = line
            steering = float(steering)

            # Perform data augmentation!

            # Center image
            frames.append(Frame(c_path, steering))
            frames.append(Frame(c_path, steering * -1, flip=True))

            # Left image
            l_steering = steering + 0.2
            frames.append(Frame(l_path, l_steering))
            frames.append(Frame(l_path, l_steering * -1, flip=True))

            # Center image
            r_steering = steering - 0.2
            frames.append(Frame(r_path, r_steering))
            frames.append(Frame(r_path, r_steering * -1, flip=True))

    random.shuffle(frames)
    return frames


def frames_to_data(frames, augment=True):
    '''
    Given frames, return features and labels
    '''
    input_data = []
    labels = []
    for frame in frames:
        input_data.append(frame.img)
        labels.append(frame.angle)

    input_data = np.asarray(input_data)
    labels = np.asarray(labels)
    return input_data, labels


def generator_training_data(frames, batch_size=BATCH_SIZE):
    '''
    Returns the Frames as training data split between test and validation
    '''
    print('Getting training data')

    index = 0
    while True:
        start = index % len(frames)
        end = min(start + batch_size, len(frames) - 1)
        input_data, labels = frames_to_data(frames[start: end])
        index += batch_size

        yield shuffle(input_data, labels)


def train():
    '''
    Trains and creates a neural network to mimic my (not-so-hot) driving
    '''
    # split data into training, validation, and testing
    training_frames, test_frames = train_test_split(get_frames(),
                                                    test_size=0.2)
    validation_frames, test_frames = train_test_split(test_frames,
                                                      test_size=0.5)

    print('Number of training frames: #{}'.format(len(training_frames)))
    print('Number of validation frames: #{}'.format(len(validation_frames)))
    print('Number of test frames: #{}'.format(len(test_frames)))

    training_generator = generator_training_data(training_frames)
    validation_generator = generator_training_data(validation_frames)
    test_generator = generator_training_data(test_frames)

    # create model
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(10, (7, 7), activation='relu'))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Convolution2D(20, (7, 7), activation='relu'))
    model.add(BatchNormalization())
    # model.add(MaxPooling2D())
    model.add(Convolution2D(48, (7, 7), activation='relu'))
    model.add(Convolution2D(60, (7, 7), activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(120))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    model.add(Dense(100))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    model.add(Dense(1))

    # Compile model
    model.compile(loss='mse', optimizer='adam')
    model.summary()

    callbacks = [
        ModelCheckpoint('models/mimic_{epoch}.h5', period=1)
    ]

    validation_input, validation_labels = frames_to_data(validation_frames)
    # Fit the model
    model.fit_generator(
        training_generator,
        epochs=20,
        steps_per_epoch=64,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=3,
        verbose=1)

    # evaluate the model
    scores = model.evaluate_generator(test_generator, steps=5)
    print('Score: {}'.format(scores))

    model.save('model.h5')


if __name__ == '__main__':
    train()
