# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./image/center.jpg
[image2]: ./image/left.jpg
[image3]: ./image/right.jpg
[image4]: ./image/centerFlip.jpg

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, 
and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network.
The model includes ReLU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.
For detail, see "2. Final Model Architecture".

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 133). 
And, the model contains batch normalization layers(model.py lines 109, 113, 117, 121, 125, 129). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 32, 90, 91).
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 139).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used center, left and right cameras when driving center lane. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to make non-underfitting and non-overfitting model.

My first step was to use a convolution neural network model with some dropout and batch normalization for reducing overfitting. 
I thought this model might be appropriate because both the training loss and validation loss are small value, and the traing time is reasonable.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

The final step was to run the simulator to see how well the car was driving around track one.
Because slow steering change when the vehicle left from the center of the load, I changed the correction value for left and right camera data from 0.2 to 0.3.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 97-135) consisted of a convolution neural network with the following layers and layer sizes.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| (1)Input         		| 160x320x3 RGB image   							| 
| (2)Lambda(Normalization)         		| divided by 255.0 and subtracted 0.5   							| 
| (3)Cropping2D     	| Cut the image. 160x320x3 -> 90x320x3 	|
| (4)Convolution 7x7					| activation:ReLU, 90x320x3 -> 84x314x16												|
| (5)Batch Normalization	      	|  				|
| (6)Max pooling	      	| 2x2 stride. 84x314x16 -> 42x157x16 				|
| (7)Batch Normalization	      	|  				|
| (8)Convolution 7x7			| activation:ReLU, 42x157x16 -> 36x151x16												|
| (9)Batch Normalization	      	|  				|
| (10)Max pooling	      	| 2x2 stride. 36x151x6 -> 18x75x16 				|
| (11)Batch Normalization	      	|  				|
| (12)Convolution 7x7			| activation:ReLU, 18x75x16 -> 12x69x16												|
| (13)Batch Normalization	      	|  				|
| (14)Convolution 9x9			| activation:ReLU, 12x69x16 -> 4x61x32												|
| (15)Batch Normalization	      	|  				|
| (16)Max pooling	      	| 2x2 stride. 4x61x32 -> 2x30x32 				|
| (17)Faltten	      	| 2x30x32 -> 1920				|
| (18)dropout		| keep prob 0.5        									|
| (19)Fully connected		| activation:ReLU, 1920 -> 32        									|
| (20)Fully connected		| activation:Linear, 32 -> 1        									|


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I used left camera and right camera when driving center lane too.

![alt text][image2]
![alt text][image3]

And, I recorded one lap for reverse driving on track one to get more data points.
Then I repeated this process on track two in order to get more data points, for one lap.

To augment the data sat, I also flipped images and angles to generalize the dataset.
For example, here is an image that has then been flipped:

![alt text][image1]
![alt text][image4]

Futhermore, I recorded the bridge driving data additionally because the difficulty of autonomous driving on the bridge.

After the collection process, I had 27543 number of data points. 
I then preprocessed this data by divided by 255.0 and subtracted 0.5.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. 
The validation set helped determine if the model was over or under fitting. 
The enough number of epochs is 10 because both training loss and validation loss changed slowly.
I used an adam optimizer so that manually training the learning rate wasn't necessary.

