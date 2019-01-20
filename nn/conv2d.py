import numpy
import math

class Conv2D:

    #input_size = (channels, height, width)
    #kernel_size = (num_filters, height, width)
    def __init__(self, input_size=(1, 28, 28), kernel_size=(3, 5, 5), learning_rate=0.1, momentum=0.5, stride=1, padding=0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.stride = stride
        self.padding = padding
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.kernel = numpy.random.uniform(-1, 1, kernel_size)
        self.bias = numpy.random.uniform(-1, 1, kernel_size[0])

        self.prev_input = None
        self.prev_weight_error = 0
        self.prev_bias_error = 0

        out_f = kernel_size[0]
        out_c = input_size[0]
        out_h = math.floor(((input_size[1] + (2 * padding)) - kernel_size[1])/stride)+1
        out_w = math.floor(((input_size[2] + (2 * padding)) - kernel_size[2])/stride)+1

        #output_size = (filters, input_channels, height, width)
        self.output_size = (out_f, out_c, out_h, out_w)
    
        self.clear_gradients()
    
    def forward(self, input):
        self.prev_input = input

        output = numpy.zeros(self.output_size)
        num_channels = self.input_size[0]
        num_filters = self.kernel_size[0]

        #iterate over each filter
        for f in range(num_filters):
            filter = self.kernel[f]
            bias = self.bias[f]

            #iterate over input channels
            for in_c in range(num_channels):
                curr_in_channel = numpy.pad(input[in_c], self.padding, 'constant')
                correlation = self.cross_correlate(curr_in_channel, filter, output[f][in_c].shape, self.stride)
                output[f][in_c] = correlation + bias
        
        return output
    
    def backward(self, a_error):
        #w_error = This vector will hold the errors for each weight in our filters
        w_error = numpy.zeros(self.kernel_size)

        #b_error = This vector will hold the errors for each bias
        b_error = numpy.zeros(self.kernel_size[0])

        #input_error = This vector will hold the errors for each input and will be
        #back propogated down to the previous layer.
        input_error = numpy.zeros(self.input_size)

        num_channels = self.input_size[0]
        num_filters = self.kernel_size[0]

        #iterate over each filter
        for f in range(num_filters):
            filter = self.kernel[f]

            #a_error_filter = This contains all of the errors associated with this filter.
            #Note that the first index into this array is for each input channel.
            a_error_filter = a_error[f]

            #w_filter_error.shape = num_channels, kernel_height, kernel_width
            w_filter_error = numpy.zeros((num_channels, self.kernel_size[1], self.kernel_size[2]))

            #iterate over input channels
            for in_c in range(num_channels):
                #a_error_channel_filter = This contains the errors associated with the current filter
                #and the current channel.
                a_error_channel_filter = a_error_filter[in_c]

                #curr_in_channel = This is the input passed during the most recent call to forward()
                #indexed for the current channel and with the appropriate padding applied.
                curr_in_channel = numpy.pad(self.prev_input[in_c], self.padding, 'constant')
                
                #The error for a filter, and each weight, is calculated by:
                #dC/dW = I * dC/dA
                #Where 'I' is the previous input and dC/dA is the error associated with each 
                #activation node, provided in a_error. The result will be an array of
                #the same shape as our filter.
                w_filter_error[in_c] = self.cross_correlate(curr_in_channel, a_error_channel_filter, w_filter_error[in_c].shape, self.stride)

                #The error for each input node is calculated by:
                #dC/dI = W * dC/dA
                #Where 'W' is our weight matrix (i.e. filter) and dC/dA is the error 
                #associated with each activation node, provided in a_error. The result 
                #will be an array of the same shape as our input.
                #Note: Prior to performing the correlation the error matrix must be 
                #rotated by 180 degrees, effectively performing a convoution.
                a_error_filter_channel_rot = numpy.rot90(numpy.rot90(a_error_channel_filter))
                filter_padding_y = a_error_filter_channel_rot.shape[0]-1
                filter_padding_x = a_error_filter_channel_rot.shape[1]-1
                padded_filter = numpy.pad(filter, ((filter_padding_y, filter_padding_y), (filter_padding_x, filter_padding_x)), mode='constant')
                input_channel_error = self.cross_correlate(padded_filter, a_error_filter_channel_rot, input_error[in_c].shape, 1)
                input_error[in_c] = input_error[in_c] + input_channel_error
            
            #The error for the bias associated with this filter is just a 
            #flattened out sum of all the activation errors associated
            #with this filter. Result is a single numeric value.
            b_error[f] = numpy.sum(a_error_filter)

            #The final error for this filter and it's weights is a sum of 
            #the calculated errors across all input channels.
            w_error[f] = numpy.sum(w_filter_error, axis=0)
        
        
        #Store the errors for the weights so we can do weight updates later.
        self.w_error_history.append(w_error)
        
        #Store the errors for the biases so we can do weight updates later.
        self.b_error_history.append(b_error)

        #Back propogate the error for our inputs
        return input_error

    def update_weights(self):
        if len(self.w_error_history) > 0:
            #dw = LR * dC/w
            #This is a vector of changes to be made to each weight scaled by
            #our learning rate.
            weight_error = (self.momentum * self.prev_weight_error) + numpy.mean(self.w_error_history, axis=0)
            self.kernel = self.kernel - (self.learning_rate * weight_error)
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
    
    #output_shape = (h, w)
    def cross_correlate(self, target, filter, output_shape, stride=1):
        output = numpy.zeros(output_shape)

        for out_y in range(output.shape[0]):
            y = out_y * stride

            for out_x in range(output.shape[1]):
                x = out_x * stride
                
                slice_y_end = y + filter.shape[0]
                slice_x_end = x + filter.shape[1]
                target_slice = target[y:slice_y_end, x:slice_x_end]
                output[out_y][out_x] = numpy.sum(target_slice * filter)
        
        return output
    
    def print_weights(self):
        print('<conv2d>')
        print(numpy.mean(self.kernel))
        print('</conv2d>')

