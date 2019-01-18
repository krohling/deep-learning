import pickle

class Network:

    def save(self, filename):
        f = open(filename,'wb') 
        pickle.dump(self, f)
        f.close()

    @staticmethod
    def open(filename):
        f = open(filename, 'rb')  
        network = pickle.load(f)
        f.close()
        
        return network
        