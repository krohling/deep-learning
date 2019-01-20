import numpy
import pickle
from nn import network, activation, conv2d, dense, reshape, maxpool2d, relu_layer

class ConvNetwork(network.Network):

    def __init__(self, image_channels, image_size, filter_count, filter_size, output_features, learning_rate=0.1, momentum=0.5):
        conv_input_size = (image_channels, image_size[0], image_size[1])
        conv_filter_size = (filter_count, filter_size[0], filter_size[1])

        self.layer1 = conv2d.Conv2D(conv_input_size, conv_filter_size, learning_rate, momentum)
        self.layer2 = relu_layer.ReluLayer()
        layer3_input_size = (self.layer1.output_size[0]*self.layer1.output_size[1], self.layer1.output_size[2], self.layer1.output_size[3])
        self.layer3 = reshape.Reshape(self.layer1.output_size, layer3_input_size)
        self.layer4 = maxpool2d.MaxPool2D(layer3_input_size, (3, 3))

        self.layer5 = conv2d.Conv2D(self.layer4.output_size, conv_filter_size, learning_rate, momentum)
        self.layer6 = relu_layer.ReluLayer()
        layer5_input_size = (self.layer5.output_size[0]*self.layer5.output_size[1], self.layer5.output_size[2], self.layer5.output_size[3])
        self.layer7 = reshape.Reshape(self.layer5.output_size, layer5_input_size)
        self.layer8 = maxpool2d.MaxPool2D(layer5_input_size, (3, 3))


        dense_input_size = self.layer8.output_size[0]*self.layer8.output_size[1]*self.layer8.output_size[2]
        self.layer9 = reshape.Reshape(self.layer8.output_size, dense_input_size)

        self.layer10 = dense.Dense(dense_input_size, output_features, activation.Sigmoid(), learning_rate, momentum)
    
    def forward(self, input):
        x = self.layer1.forward(input)
        x = self.layer2.forward(x)
        x = self.layer3.forward(x)
        x = self.layer4.forward(x)
        x = self.layer5.forward(x)
        x = self.layer6.forward(x)
        x = self.layer7.forward(x)
        x = self.layer8.forward(x)
        x = self.layer9.forward(x)
        return self.layer10.forward(x)

    def backward(self, criterion, output, target):
        local_error = criterion.d_cost(output, target)
        local_error = self.layer10.backward(local_error)
        local_error = self.layer9.backward(local_error)
        local_error = self.layer8.backward(local_error)
        local_error = self.layer7.backward(local_error)
        local_error = self.layer6.backward(local_error)
        local_error = self.layer5.backward(local_error)
        local_error = self.layer4.backward(local_error)
        local_error = self.layer3.backward(local_error)
        local_error = self.layer2.backward(local_error)
        self.layer1.backward(local_error)
    
    def udpate_weights(self):
        self.layer10.update_weights()
        self.layer9.update_weights()
        self.layer8.update_weights()
        self.layer7.update_weights()
        self.layer6.update_weights()
        self.layer5.update_weights()
        self.layer4.update_weights()
        self.layer3.update_weights()
        self.layer2.update_weights()
        self.layer1.update_weights()
    
    def print_weights(self):
        print("<conv_network>")
        self.layer10.print_weights()
        self.layer9.print_weights()
        self.layer8.print_weights()
        self.layer7.print_weights()
        self.layer6.print_weights()
        self.layer5.print_weights()
        self.layer4.print_weights()
        self.layer3.print_weights()
        self.layer2.print_weights()
        self.layer1.print_weights()
        print("</conv_network>")
    
    def save(self, filename):
        f = open(filename,'wb') 
        pickle.dump(self, f)
        f.close()

    @staticmethod
    def open(filename):
        f = open(filename, 'r')  
        network = pickle.load(f)
        f.close()
        
        return network
