# Neural-Network

This is a simple NN project written in Matlab (without a library) with only one hidden layer. 
It is for educational purpose, it was a university project in Machine_Learning.

The NN detect handwritten numbers which on mnist_data.

The main script is the 'demo_mnistLogreg.m' file which loads the data from the mnist_all.mat, split them to training_set and test_set, normalize the pixels to take values in [0,1] , and calls the other main functions such as costgrad_softmax, ml_softmax , etc.

The main purpose is to detect the handwritten numbers and this will done with multiple logistic regression which will classify the data we have. The idea is to take a large number of handwritten digits, known as training examples, and then develop a system which can learn from those training examples. In other words, the neural network uses the examples to automatically infer rules for recognizing handwritten digits.

The data we have looks like... :

![alt tag](http://neuralnetworksanddeeplearning.com/images/mnist_100_digits.png)


To recognize individual digits we will use a three-layer neural network. 

* Bullet list The first layer is the input layer, the second is the hidden layer and the third is the output layer. The input layer of the network contains neurons encoding the values of the input pixels. 

* Bullet list The second layer of the network is a hidden layer. We denote the number of neurons in this hidden layer by nn, and we'll experiment with different values for n.

* Bullet list The output layer of the network contains 10 neurons. If the first neuron fires, i.e., has an output ≈1≈1, then that will indicate that the network thinks the digit is a 00. If the second neuron fires then that will indicate that the network thinks the digit is a 11. And so on. A little more precisely, we number the output neurons from 00 through 99, and figure out which neuron has the highest activation value. If that neuron is, say, neuron number 66, then our network will guess that the input digit was a 6. And so on for the other output neurons.

This is an image of NN structure :

![alt tag](http://neuralnetworksanddeeplearning.com/images/tikz12.png)

*the hidden units in this image are 15 but the default number of hidden units in the project is 250*


