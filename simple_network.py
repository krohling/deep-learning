import numpy
import pickle
from nn import network, activation, conv2d, dense, reshape

class SimpleNetwork(network.Network):

    def __init__(self, learning_rate=0.1):
        self.layer1 = dense.Dense(2, 10, activation.Relu(), learning_rate)
        self.layer2 = dense.Dense(10, 1, activation.Sigmoid(), learning_rate)
    
    def forward(self, input):
        x = self.layer1.forward(input)
        return self.layer2.forward(x)

    def backward(self, criterion, output, target):
        local_error = criterion.d_cost(output, target)
        local_error = self.layer2.backward(local_error)
        self.layer1.backward(local_error)
    
    def udpate_weights(self):
        self.layer2.update_weights()
        self.layer1.update_weights()
        