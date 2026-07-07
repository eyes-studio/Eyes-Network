import math
import random
#math core
class MathENG1:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    
    @staticmethod
    def relu(x):
        return max(0, x)
    
    @staticmethod
    def gelu(x):
        return 0.5 * x * (
            1 + math.tanh(
                math.sqrt(2 / math.pi) *
                (x + 0.044715 * x**3)
            )
        )
    
    @staticmethod
    def sigmoid_derivative(x):
        y = 1 / (1 + math.exp(-x))
        return y * (1 - y)
    
    @staticmethod
    def relu_derivative(x):
        if x > 0:
            return 1
        else:
            return 0
    
    @staticmethod
    def gelu_derivative(x):
        a = math.sqrt(2 / math.pi)
    
        tanh_part = math.tanh(
            a * (x + 0.044715 * x**3)
        )

        sech = 1 - tanh_part**2

        return 0.5 * (
            1 + tanh_part +
            x * sech * a * (1 + 3 * 0.044715 * x**2)
        )
    
    @staticmethod
    def mse_loss(output, target):
        return sum((o - t)**2 for o, t in zip(output, target)) / len(output)
    
#ai core
class Neuron:
    def __init__(self, inputs,activition='gelu'):
        self.i=inputs
        self.a=activition
        self.multiplier=random.uniform(-0.5,0.5)
        self.addition=0

    def forward(self,inputs):
        lens = len(inputs)
        if self.a == 'gelu':
            out=[]
            for i in range(lens):
                inp=inputs[i]
                out.append(MathENG1.gelu(inp * self.multiplier + self.addition))
            return out
        elif self.a == 'relu':
            out=[]
            for i in range(lens):
                inp=inputs[i]
                out.append(MathENG1.relu(inp * self.multiplier + self.addition))
            return out
        elif self.a == 'sigmoid':
            out=[]
            for i in range(lens):
                inp=inputs[i]
                out.append(MathENG1.sigmoid(inp * self.multiplier + self.addition))
            return out
        else:
            print("SYNTEX EROR")
            return 
    
    def train(self,inputs,output,learning_rate):
        otput_now = self.forward(inputs)

        for i in range(len(output)):
            eror = output[i] - otput_now[i]

            self.multiplier += eror * learning_rate
            self.addition += eror * learning_rate
        
        

class Layer:
    def __init__(self, input_size, neuron_count):
        self.si=input_size
        self.nc=neuron_count

    def forward(self):
        pass


class Network:
    def __init__(self,neoron_inputs,layers,neoron_on_layers,output_neoron,activation='gelu'):
        self.ni=neoron_inputs
        self.l=layers
        self.nol=neoron_on_layers
        self.on=output_neoron
        self.ac=activation

    def forward(self):
        pass