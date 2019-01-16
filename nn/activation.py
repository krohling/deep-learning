import math

class Sigmoid:

    def fx(self, x):
        return 1 / (1 + math.exp(-x))
    
    def dfx(self, x):
        val = self.fx(x)
        return val*(1-val)

class Relu:
    def fx(self, x):
        if x > 0:
            return x
        else:
            return 0
    
    def dfx(self, x):
        return 1. * (x > 0)