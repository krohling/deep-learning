import numpy
from conv_network import ConvNetwork
from nn import error
from random import shuffle

EPOCH_COUNT = 20
IMAGE_CHANNELS = 1
IMAGE_SIZE = (28, 28)
CONV_FILTER_COUNT = 5
CONV_FILTER_SIZE = (3, 3)
OUTPUT_FEATURES = 10
LEARNING_RATE = 0.1
BATCH_SIZE = 250
NETWORK_FILENAME = 'conv_test.net'

network = ConvNetwork(IMAGE_CHANNELS, IMAGE_SIZE, CONV_FILTER_COUNT, CONV_FILTER_SIZE, OUTPUT_FEATURES, LEARNING_RATE)
criterion = error.MSE()

def parse_minst_dataset(path):
    f  = open(path, "r")
    data = f.read()
    lines = data.split('\n')
    result = []

    for line in lines:
        cols = line.split(',')
        target = numpy.zeros(10)
        target[int(cols[0])] = 1
        input = cols[1:785]
        input = numpy.array(list(map(int, input)))
        input = input/255
        input = numpy.reshape(input, (28, 28))
        input = numpy.array([input])
        result.append((input, target))
    
    return numpy.array(result)

print("parse training")
training_set = parse_minst_dataset("mnist/mnist_train.csv")
shuffle(training_set)
print("parse validation")
validation_set = parse_minst_dataset("mnist/mnist_test.csv")
shuffle(validation_set)
display_set = validation_set[0:10]

print("start training")
network.print_weights()
for epoch in range(EPOCH_COUNT):
    training_error = 0
    training_count = 0
    for input, target in training_set:
        output = network.forward(input)
        network.backward(criterion, output, target)
        training_error += criterion.cost(output, target)

        training_count += 1
        if training_count % BATCH_SIZE == 0:
            print("Batch Update - Count: %i" % (training_count))
            network.udpate_weights()
            network.print_weights()

    network.udpate_weights()

    for input, target in display_set:
        output = network.forward(input)
        pred = numpy.argmax(output)
        truth = numpy.argmax(target)
        
        print("********")
        print(output)
        print(target)
        print(pred)
        print(truth)

    validation_error = 0
    validation_correct = 0
    for input, target in validation_set:
        output = network.forward(input)
        validation_error += criterion.cost(output, target)

        pred = numpy.argmax(output)
        truth = numpy.argmax(target)
        if pred == truth:
            validation_correct += 1

    training_error = training_error/len(training_set)
    validation_error = validation_error/len(validation_set)
    validation_acc = validation_correct/len(validation_set)

    network.save(NETWORK_FILENAME)
    print("Epoch: %i Training Error: %.6f Validation Error: %.6f Validation Acc: %.6f" % (epoch, training_error, validation_error, validation_acc))

