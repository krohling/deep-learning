import numpy

class MSE:

    def cost(self, outputs, targets):
        cost = numpy.subtract(targets, outputs)
        cost = numpy.square(cost)
        return numpy.sum(cost)/len(targets)
    
    def d_cost(self, outputs, targets):
        return outputs - targets

class SSE:

    def cost(self, outputs, targets):
        cost = numpy.subtract(outputs, targets)
        cost = numpy.square(cost)
        return numpy.sum(cost)
    
