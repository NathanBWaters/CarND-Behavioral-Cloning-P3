# **Behavioral Cloning** 


![self-driving-car](self_driving.gif) 

This project uses the Udacity car simulator in order to train a model to learn
how to drive around a track.  The input is the footage of the car driving and
the output of the model is the direction of the steering wheel.  I generated
about 0.9 GB of training data of me driving around the track.  This also
contained footage of the car getting back on the road and staying center.

All of my work is done in `model.py`.

#### 1. Architecture


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

