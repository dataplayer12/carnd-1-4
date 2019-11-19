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

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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

**Background:** As a deep learning practitioner, I have realized that optimizing the architecture of the neural network is often more important than collecting more training data. Of course, training data should be enough to let the model generalize, but we often run into diminishing returns as we add data beyond a certain point. With this in mind, I collected training data by driving the car thrice around the course, but did not perform some of the more sophisticated data collection strategies outlined in the project, such as recording recoveries from teh sides of roads. Most of the thrust of my work was thus, in optimizing the architecture of the neural network. The architecture used in this project is inspired by a [paper](https://arxiv.org/abs/1412.6806) from Springenberg et. al., who proposed doing away with many dense layers at the end of a classifier and using convolutional layers as much as possible for modelling as well as downsampling. This reduces the number of parameters of the network and prevents overfitting usually associated with dense layers. Their paper inspired many successful architectures like VGG and ultimately ResNet. Thus, I found this architecture to be a good starting point for this project. Another paper used for designing this model was on [Wide ResNets](https://arxiv.org/abs/1605.07146) by Zagoruyko et al., who showed that wide CNNs with many filters in convlutional layers are more effective at classification tasks than deeper networks with fewer filters in their convolutional layers.

##### Architecture
- The model starts out by cropping the input images to remove the sky and trees as well as the front hood of the car. The resulting image is 90x320x3. (line 72)
- The images are then normalized to 0 mean with a Lambda layer from keras. (line 73)
- The normalized images are fed into a convolution block which has a kernel size of `5x5` and `16` filters, followed by and relu activation. This returns a tensor of size `45x160x16`. (line 76)
- The next layer is a convolution block which has a kernel size of `5x5` and `32` filters, followed by and relu activation. This returns a tensor of size `23x80x32`. (line 79)
- The next layer is again a convolution block which has a kernel size of `5x5` and `64` filters, followed by and relu activation. This returns a tensor of size `12x40x64`. (line 82)
- This is followed by a convolutional layer which has a kernel size of `5x5` and `128` filters, followed by and relu activation. This returns a tensor of size `6x20x128`. (line 85)
- The last convolutional layer in this model has a kernel size of `3x3` and `256` filters, followed by and relu activation. This returns a tensor of size `3x10x256`. (line 88)
- Next we flatten the output of the previous layer into a 1-D tensor of size `7680`. (line 91)
- We want to add a dense layer. In order to prevent overfitting, we add a dropout layer with a drop probability of 0.3. (line 94)
- After dropout, we add a dense layer with `256` neurons. (line 97)
- Finally we have a dense layer with one neuron predicting the driving command of the car. Importantly, we use `tanh` activation in this last layer to scale the output in the range [-1,1]. (line 100) The activation `tanh` allowed the model to make driving predictions which were realistic. The `tanh` also smoothed large gradients, especially at the beginning of the training, which were causing the model to go haywire.

#### 2. Attempts to reduce overfitting in the model

We have tried to minimize the number of dense layers in the architecture to reduce overfitting. The model contains one dropout layer with a drop probability of `0.3` to reduce overfitting. (line 94). We found that the default value of `0.5` was too high for the model to learn quickly. Reducing it to `0.3` helped stabilize the car around sharp corners of the course.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 65, 66, 106). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 103).
Several batch sizes were tried but a size of `32` provided a good balance between training time and low validation loss.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Training data was collected by driving the car around the course three times. Having collected enough training data, we used the outputs of all the cameras, left, right and center to simulate recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I have explained my motivations behind the design of the convlutional network in detail at the beginning of this write-up in `Background`. To summarize, I found that adding more data was leading to diminishing returns. Thus, I put most of my efforts in imporving the architecture of the CNN and reduce overfitting. Two papers ([All ConvNets](https://arxiv.org/abs/1412.6806) & [Wide ResNets](https://arxiv.org/abs/1605.07146)) were used as guidelines for designing a model with no pooling layers, just one dense layer (except the output layer) and wide convolutional blocks to do the bulk of heavy-lifting.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. This was because I had been using only the center images as input to the network which resulted in the netowrk being severely overparametrized. Adding the left and right camera angles reduced overiftting significantly but introduced a new parameter, called `angle_correction` in the code (line 24), which had to be tuned. I tried several values like 0.2, 0.1, 0.25, 0.35, but in these cases, the car either didn't react strongly enough to the bends in the road or reacted too sharply causing it to go overboard. Finally a good value of `0.3` was obtained after several trials to be the optimal value for my training dataset.

I also used slightly more epochs than necessary (8 epochs). Although the model trained well in 5 epochs I used more epochs to verify that the validation loss did not fluctuate too much which would indicate that the model had found a shallow local minima and would not generalize well. The training runs verified that the validation loss was stable for 2-3 epochs and thus the model was not stuck in a local minima. These optimizations led to a model with the lowest validation loss among all my trials.

The final step was to run the simulator to see how well the car was driving around track one. To my delight, the model performed very well and the car was able to traverse the course on its own.

#### 2. Final Model Architecture

The final model architecture (model.py lines 71-100) consisted of a convolution neural network with 5 convolutional layers, a dense layer, a dropout layer and a final output layer with tanh activation. A detailed description of the model architecture along with input and output shapes and line numbers in `model.py` was provided in the `Architecture` section.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

As mentioned earlier, I found that recording recoveries of the car from either lanes led to diminishing returns. Thus, I focused my efforts on imporving the architecture.

To augment the data sat, I used the left and right camera images of the training data to simulate the car going off-track.

In the model architecture, I cropped the images to remove trees and car body. Althought e cropping parameters for the left, right and center images were different, there was no way to specify this in the keras model. I chose parameters which worked well enough for all, but this is a potential area for improvement in this model.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as the model achieved a very low validation loss. However, I used more epochs (8 epochs) in order to verify that the validation loss was indeed stable and that the minima found by the optimizer was sufficiently broad so as to generalize successfully. I used an adam optimizer so that manually training the learning rate wasn't necessary.
