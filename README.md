# CatsvsDogs
# Introduction
This model is for the classification of cats vs dogs . I have used a simple CNN architecture in this model containig 2 Convolution layers followed by Max pooling layer. Then i have used 2 fully connected layer. Adam optimzer is used in this model for better optimization.Epochs used is 50.
# Files
Image.py contains the code of the classifier.
my_model.h5 contains the trained model.
my_mode_weights.h5 contains the trained weights.
1.py contains loaded model and use for predicting the class (0/1) dogs and cats
# Library used
keras.
keras.model for importing Sequential model.
keras.layers for importing Dense,Conv2D,Maxpool layers,Flatten.
numpy,matplotlib.pyplot,
# Model 
This model contains total 4 layers and 2 max pool layers. First layer is convolution layer with 128 filters size (5,5) along with the bias value and the input is (64,64,3) 3 is for RBG . Strides use is 1 and padding is 'valid'(output is less than input features). Activation function used is relu . Followed by maxpool layer with size (3,3) and padding valid . Second layer is also Convolution layer with 64 filter and all the things like padding , stride is of same size but size of filters is now (3,3).Again followed by maxpool layer . Now a fuction is used for flatting the output from maxpool layer into a vector which goes into the fully connected layer with 128 nodes and activation used is relu. Last layer used contains only one node for (0/1) with activation function sigmoid.

