import numpy
from nn import activation

class ReluLayer:

    def __init__(self):
        self.relu = activation.Relu()
        self.activation_fx = numpy.vectorize(self.relu.fx)
        self.activation_dfx = numpy.vectorize(self.relu.dfx)
    
    def forward(self, input):
        self.prev_activation_dfx = self.activation_dfx(input)
        return self.activation_fx(input)
    
    def backward(self, d_cost):
        return d_cost * self.prev_activation_dfx
    
    def update_weights(self):
        pass

    def print_weights(self):
        pass