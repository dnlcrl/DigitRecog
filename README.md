# DigitRecog
A neural network for recognizing digits 

# Requisites
- NumPy:
    
    pip install numpy

# Usage

cd in the src folder, launch python compiler and:
 

    >>> import mnist_loader
    >>> training_data, validation_data, test_data = \ ... mnist_loader.load_data_wrapper()

    >>> import network
    >>> net = network.Network([784, 30, 10]) # optimal basic architecture

    >>> net.SGD(training_data, 30, 10, 3.0, test_data=test_data) # optimal configuration

check the code docs for more infos

  