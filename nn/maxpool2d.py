import numpy
import math

class MaxPool2D:

    #input_size = (channels, height, width)
    #stride = (y, x)
    def __init__(self, input_size, kernel_size=(2, 2), stride=None, padding=0):
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.padding = padding

        if(stride is None):
            #Stride is assumed to be the size of the kernel
            self.stride = kernel_size
        else:
            self.stride = stride

        out_c = input_size[0]
        out_h = math.floor(((input_size[1] + (2 * padding)) - kernel_size[0])/self.stride[0])+1
        out_w = math.floor(((input_size[2] + (2 * padding)) - kernel_size[1])/self.stride[1])+1
        self.output_size = (out_c, out_h, out_w)
    
    def forward(self, input):
        output = numpy.zeros(self.output_size)
        num_channels = self.output_size[0]
        out_height = self.output_size[1]
        out_width = self.output_size[2]
        kernel_height = self.kernel_size[0]
        kernel_width = self.kernel_size[1]
        stride_y = self.stride[0]
        stride_x = self.stride[1]

        prev_input_indices = numpy.zeros((num_channels, out_height, out_width, 2))

        for in_c in range(num_channels):
            curr_in_channel = numpy.pad(input[in_c], self.padding, 'constant')
            
            for out_y in range(out_height):
                in_y = out_y * stride_y

                for out_x in range(out_width):
                    in_x = out_x * stride_x
                    
                    slice_y_end = in_y + kernel_height
                    slice_x_end = in_x + kernel_width
                    target_slice = curr_in_channel[in_y:slice_y_end, in_x:slice_x_end]

                    max_index = numpy.unravel_index(target_slice.argmax(), target_slice.shape)
                    prev_input_indices[in_c][out_y][out_x][0] = max_index[0] + in_y - self.padding
                    prev_input_indices[in_c][out_y][out_x][1] = max_index[1] + in_x - self.padding
                    output[in_c][out_y][out_x] = target_slice[max_index]
        
        self.prev_input_indices = prev_input_indices

        return output
    
    def backward(self, a_error):
        input_error = numpy.zeros(self.input_size)

        num_channels = self.output_size[0]
        error_height = self.output_size[1]
        error_width = self.output_size[2]
        input_height = self.input_size[1]
        input_width = self.input_size[2]

        for in_c in range(num_channels):
            for a_error_y in range(error_height):
                for a_error_x in range(error_width):
                    input_error_y = int(self.prev_input_indices[in_c][a_error_y][a_error_x][0])
                    input_error_x = int(self.prev_input_indices[in_c][a_error_y][a_error_x][1])

                    if input_error_y >= 0 and input_error_x >=0 and input_error_y < input_height and input_error_x < input_width:
                        input_error[in_c][input_error_y][input_error_x] = a_error[in_c][a_error_y][a_error_x]
        
        return input_error


        
    
    def update_weights(self):
        pass
    
    def clear_gradients(self):
        pass

    def print_weights(self):
        pass