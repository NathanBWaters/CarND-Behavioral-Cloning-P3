# **Behavioral Cloning** 

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Here is the architecture of my model:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 59, 314, 10)       1480
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 29, 157, 10)       0
_________________________________________________________________
batch_normalization_1 (Batch (None, 29, 157, 10)       40
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 23, 151, 20)       9820
_________________________________________________________________
batch_normalization_2 (Batch (None, 23, 151, 20)       80
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 17, 145, 48)       47088
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 11, 139, 60)       141180
_________________________________________________________________
batch_normalization_3 (Batch (None, 11, 139, 60)       240
_________________________________________________________________
flatten_1 (Flatten)          (None, 91740)             0
_________________________________________________________________
dropout_1 (Dropout)          (None, 91740)             0
_________________________________________________________________
dense_1 (Dense)              (None, 120)               11008920
_________________________________________________________________
batch_normalization_4 (Batch (None, 120)               480
_________________________________________________________________
dense_2 (Dense)              (None, 100)               12100
_________________________________________________________________
batch_normalization_5 (Batch (None, 100)               400
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 101
=================================================================
```

I was not able to make a larger model due to memory exhaustion errors on my Nvidia card.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.  It also contains BatchNormalization layers after the CNN and Dense layers to normalize the data.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used the Adam Optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I manually generated about 0.9 GB of training data.  Most of it was simply driving through while staying in the center lane.  I also generated some examples of the car starting off too far off to the left or right side of the lane and re-centering.  

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to replicate the Nvidia architecture.  I moved away from this architecture though because it was simply too big for my GPU to handle.

Moving forward, I followed the pattern of gradually moving thinner and deep of a network by moving through a succession of Convolution2D and MaxPooling layers.

In order to gauge how well the model was working, I split my image and steering angle data into a training, validation, and test set.  I noticed that both the training and validation loss was decreasing, so I increased the number of epochs from 10 to 20.

To combat the overfitting, I added BatchNormalization and Dropout layers.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 59, 314, 10)       1480
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 29, 157, 10)       0
_________________________________________________________________
batch_normalization_1 (Batch (None, 29, 157, 10)       40
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 23, 151, 20)       9820
_________________________________________________________________
batch_normalization_2 (Batch (None, 23, 151, 20)       80
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 17, 145, 48)       47088
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 11, 139, 60)       141180
_________________________________________________________________
batch_normalization_3 (Batch (None, 11, 139, 60)       240
_________________________________________________________________
flatten_1 (Flatten)          (None, 91740)             0
_________________________________________________________________
dropout_1 (Dropout)          (None, 91740)             0
_________________________________________________________________
dense_1 (Dense)              (None, 120)               11008920
_________________________________________________________________
batch_normalization_4 (Batch (None, 120)               480
_________________________________________________________________
dense_2 (Dense)              (None, 100)               12100
_________________________________________________________________
batch_normalization_5 (Batch (None, 100)               400
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 101
=================================================================
```


#### 3. Creation of the Training Set & Training Process

I recorded 0.9 GB of data where I recorded myself in the following situations:
1) Normally looping around track 1
2) Normally looping around track 2
3) Starting off-center on curves and getting back in the center

I took each frame and split it into 6 pieces of data where took the original and inverted version of the left, center, and right image:
```python
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
```

This greatly augmented my dataset.  After the collection process, I had X number of data points. I then preprocessed this data by simply dividing by 255.0 and then subtracting by 0.5 so that the pixel values are centered around 0 and is normalized between -0.5 and 0.5.  I also cropped out the top and bottom of the frame in the network.

I split the total of my data between three sets: training (90%), validation (5%), and test (5%)

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.  I used an Adam optimizer so that manually training the learning rate wasn't necessary.

