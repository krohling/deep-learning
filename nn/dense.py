import numpy
import math

class Dense:

    def __init__(self, in_features, out_features, activation, learning_rate=0.1, momentum=0.5, weights=None, bias=None):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.activation_fx = numpy.vectorize(activation.fx)
        self.activation_dfx = numpy.vectorize(activation.dfx)

        self.clear_gradients()
        self.prev_weighted_sum = 0
        self.prev_activation_dfx = 0
        self.prev_weight_error = 0
        self.prev_bias_error = 0

        if weights is None:
            self.weights = numpy.random.uniform(-1, 1, (out_features, in_features))
        else:
            self.weights = weights
        
        if bias is None:
            self.bias = numpy.random.uniform(-1, 1, out_features)
        else:
            self.bias = bias
    
    def forward(self, input):
        self.prev_input = input
        weighted_sum = numpy.dot(self.weights, input) + self.bias
        self.prev_activation_dfx = self.activation_dfx(weighted_sum)

        return self.activation_fx(weighted_sum)
    
    def backward(self, a_error):
        #a_error = dC/dA
        #This is a vector containing the derivative of the 
        #cost function with respect to each node's activation.


        #z_error = dC/dZ = dC/dA * dA/dZ
        #This is a vector of the error associated with
        #each node's weighted sum (Z).
        z_error = a_error * self.prev_activation_dfx
        z_error_t = z_error[numpy.newaxis].T

        #w_error = dC/w = dC/dA * dA/dZ * dZ/w
        #This is a vector of the error associated with
        #each weight.
        w_error = (z_error_t * self.prev_input)
        
        #Store the errors for the weights so we can do weight updates later.
        self.w_error_history.append(w_error)
        
        #Store the errors for the biases so we can do weight updates later.
        self.b_error_history.append(z_error)
        
        input_error = numpy.sum(z_error_t * self.weights, axis=0)

        return input_error
    
    def update_weights(self):
        if len(self.w_error_history) > 0:
            #dw = LR * dC/w
            #This is a vector of changes to be made to each weight scaled by
            #our learning rate.
            weight_error = (self.momentum * self.prev_weight_error) + numpy.mean(self.w_error_history, axis=0)
            self.weights = self.weights - (self.learning_rate * weight_error)
            self.prev_weight_error = weight_error
        
        if len(self.b_error_history) > 0:
            #db = LR * dC/b
            #This is a vector of changes to be made to each bias scaled by
            #our learning rate.
            bias_error = (self.momentum * self.prev_bias_error) + numpy.mean(self.b_error_history, axis=0)
            self.bias = self.bias - (self.learning_rate * bias_error)
            self.prev_bias_error = bias_error
        
        self.clear_gradients()
    
    def clear_gradients(self):
        self.w_error_history = []
        self.b_error_history = []

    def print_weights(self):
        print('<dense>')
        print(numpy.mean(self.weights))
        print('</dense>')