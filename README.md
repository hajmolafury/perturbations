# perturbations
This repo is the implementation of perturbation based algorithms for training neural networks - our main interest lies in evaluating the scalability of node perturbation for hard object recognition tasks.
By default, the code trains and evaluates the network on MNIST.
master.py is the main script which creates and trains a multi-layer perceptron with perturbation based methods, which include adding gaussian white noise to the input image (input perturbation) and to the output of a node (node perturbation). One can also chose to train with a hybrid learning rule which is a mix of node perturbation and sgd. The amount of 'hybrid' can be specified by the parameter alpha, with 1 corresponding to (plain) sgd and 0 to (plain) node perturbation.
