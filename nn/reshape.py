import numpy

class Reshape:

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def forward(self, input):
        return numpy.reshape(input, self.output_shape)
    
    def backward(self, d_cost):
        return numpy.reshape(d_cost, self.input_shape)
    
    def update_weights(self):
        pass

    def print_weights(self):
        pass