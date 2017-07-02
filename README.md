# **Behavioral Cloning**
_Lorenzo's version_

---

![alt text][simul]

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model]: ./Model_Architecture.png "Model Visualization"
[simul]: ./simulator.png "Simulator"
[cameras]: ./left-center-right_samples.png "Cameras"
[flipped]: ./flipped_samples.png "Flipped"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality



My project includes the following files:
* `model.py` containing the script to create and train the model
* `model.ipynb` which is the jupyter notebook I used for saving model.py
* `drive.py` for driving the car in autonomous mode  _(note: I have changed the speed to 25 mph in this file)_
* `model.h5` containing a trained convolution neural network
* `readme.md` which is the project writeup (you are currently reading it!) summarizing the results


Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

The `model.py` file contains the code for training and saving the convolution neural network. Running it takes about 5 minutes on my GPU (GTX 960) and 25 minutes on CPU. The file shows the pipeline I used for training and validating the model, and `model.ipynb` contains comments to explain how the code works.

### Model Architecture and Training Strategy

- _An appropriate model architecture has been employed_

My model consists of a 4 convolutional layers + Max Pooling followed by 3 dense layers and the final output layer. It follows the VGGNet principles for the convolutional part with 3x3 filter sizes (with padding 1) and 2x2 Max pooling layers (no padding) (model.py lines 80-88).

The model includes RELU layers to introduce nonlinearity, the data is normalized in the model using a Keras lambda layer (code line 77), and the image is finally resized to take less memory (height and width both divided by 2, code line 78).

- _Attempts to reduce overfitting in the model_

The model contains dropout layers in order to reduce overfitting (model.py lines 135-140).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 69-74). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

- _Model parameter tuning_

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 145).

- _Appropriate training data_

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, reverse track center lane driving, 3 cameras with modified steering angle of +0.3 for the left image (closer to the left line) and -0.3 for the right image. I also added the flipped image and steering angle systematically.

For details about how I created the training data, see the next section.

- _Final model architecture_
![alt text][model]

- _Solution Design Approach_

The overall strategy for deriving a model architecture was to first start to have a working pipeline. I started with a simple LeNet architecture, which is quite fast to train, in fact  most of the complexity is in the pre-processing step, where `cv2.imread` actually reads images in _BGR_, so I had to convert them back to _RGB_. once I started to have relevant predictions (vehicle started to actually turn correctly), I move on to the current architecture.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.
To combat the overfitting, I simply added dropouts after the dense layers.

The final step was to run the simulator to see how well the car was driving around track one. I had a few iterations where I simply added more data and saw a direct improvement on how the network was driving (e.g. added one lap). But then, with that much data, I had to build a `train_generator` and `validation_generator` in order to free some memory, as well as resizing the images to 1/4.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.




### Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][cameras]

I then recorded one lap of driving in the reverse direction.


To augment the data sat, I also flipped images and angles thinking that this would naturally increase the dataset and help avoid an imbalance on average steering angles. For example, here is an image that has then been flipped:

![alt text][flipped]

After the collection process, I had approx. 40k number of data points. I then randomly shuffled the data set and put 20% of the data into a validation set.
Also, in the Keras model itself, I preprocessed this data by cropping only the part we are interesting in (i.e. taking out the top of the images with the trees and the bottom with the front of the car), normalized it using a Lambda layer, and resized to 1/4 (divided length and width by 2).

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the fact that the training and validation errors did not seem to improve past that number of epochs. As described in the model architecture above. I used an Adam optimizer so that manually training the learning rate wasn't necessary.
