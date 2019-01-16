import numpy
from simple_network import SimpleNetwork
from nn import error

EPOCH_COUNT = 500
LEARNING_RATE = 1

network = SimpleNetwork(LEARNING_RATE)
criterion = error.MSE()

training_set = [
    ([0.0, 0.0], [0]),
    ([0.0, 0.99], [0.99]),
    ([0.99, 0.0], [0.99]),
    ([0.99, 0.99], [0])
]

for i in range(EPOCH_COUNT):
    cost = 0
    count = 0
    for input, target in training_set:
        input = numpy.array(input)
        target = numpy.array(target)

        output = network.forward(input)
        network.backward(criterion, output, target)

        cost += criterion.cost(output, target)
        count += 1
    
    network.udpate_weights()
    print(cost/count)

for input, target in training_set:
    input = numpy.array(input)
    target = numpy.array(target)

    output = network.forward(input)
    print(input)
    print(output)

