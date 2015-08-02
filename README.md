# DigitRecog
A neural network for recognizing digits 

# Requisites
- NumPy:
    
    pip install numpy

# Usage

cd in the src folder, launch python compiler and:

random weight and bias initialization, Quadratic cost as cost function:

    >>> import mnist_loader
    >>> training_data, validation_data, test_data = \ ... mnist_loader.load_data_wrapper()

    >>> import network
    >>> net = network.Network([784, 30, 10]) # optimal basic architecture

    >>> net.SGD(training_data, 30, 10, 3.0, test_data=test_data) # optimal configuration

random weigth and bias initialization, Cross-entropy as cost function:

    >>> import mnist_loader
    >>> training_data, validation_data, test_data = \
    ... mnist_loader.load_data_wrapper()
    >>> import network2
    >>> net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost) 
    >>> net.large_weight_initializer()
    >>> net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data,
    ... monitor_evaluation_accuracy=True)

Overfitting:
using just the first 1,000 training images. Using that restricted set will make the problem with generalization much more evident. With a learning rate of η = 0.5 and a mini- batch size of 10, and 400 epochs because we're not using as many training examples. (the network is overfitting (or overtraining) beyond epoch 280.It's almost as though the network is merely memorizing the training set, without understanding digits well enough to generalize to the test set.)

    >>> import mnist_loader
    >>> training_data, validation_data, test_data = \
    ... mnist_loader.load_data_wrapper()
    >>> import network2
    >>> net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost) 
    >>> net.large_weight_initializer()
    >>> net.SGD(training_data[:1000], 400, 10, 0.5, evaluation_data=test_data, 
    ... monitor_evaluation_accuracy=True, monitor_training_cost=True)

Regularization:
Using a regularized cost function with λ = 0.1, Cross-entropy as cost function, 30 hidden neurons, a mini-batch size of 10, a learning rate of 0.5. This time the accuracy on the test_data continues to increase for the entire 400 epochs It seems that, empirically, regularization is causing our network to generalize better, and considerably reducing the effects of overfitting.:

    >>> import mnist_loader
    >>> training_data, validation_data, test_data = \
    ... mnist_loader.load_data_wrapper()
    >>> import network2
    >>> net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost) >>> net.large_weight_initializer()
    >>> net.SGD(training_data[:1000], 400, 10, 0.5,
    ... evaluation_data=test_data, lmbda = 0.1,
    ... monitor_evaluation_cost=True, monitor_evaluation_accuracy=True,
    ... monitor_training_cost=True, monitor_training_accuracy=True)

Finally, 100 hidden neurons and a regularization parameter of λ = 5.0, cross-entropy cost function and L2 regularization. The final result is a classification accuracy of 97.92 percent on the validation data. That's a big jump from the 30 hidden neuron case. In fact, tuning just a little more, to run for 60 epochs at η = 0.1 and λ = 5.0 we break the 98 percent barrier, achieving 98.04 percent classification accuracy on the validation data.


    >>> net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)
    >>> net.large_weight_initializer()
    >>> net.SGD(training_data, 30, 10, 0.5, lmbda=5.0,
    ... evaluation_data=validation_data,
    ... monitor_evaluation_accuracy=True)

Weight Inizializing: Network2.py uses by default a new standard deviation = sqrt(3/2) to inizialize the weights, so net.large_weight_initializer() call can be omitted

    >>> net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
    >>> net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0,
    ... evaluation_data=validation_data,
    ... monitor_evaluation_accuracy=True)



check the code docs for more infos

  