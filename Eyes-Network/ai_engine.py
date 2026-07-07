import math
import random
from other import flatten
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
    
#neoron
class Neuron:
    def __init__(self, inputs,activition='gelu'):
        self.i=inputs
        self.a=activition

        self.multiplier=[]
        for i in range(self.i):
            self.multiplier.append(random.uniform(-0.5, 0.5))
        
        self.addition=0

    def forward(self,inputs):
        if len(self.multiplier) != len(inputs):
            print("INPUTS ERROR")
            return
        lens = len(inputs)
        if self.a == 'gelu':
            total = 0

            for i in range(len(inputs)):
                total += inputs[i] * self.multiplier[i]

            return MathENG1.gelu(total + self.addition)
        elif self.a == 'relu':
            total = 0

            for i in range(len(inputs)):
                total += inputs[i] * self.multiplier[i]

            return MathENG1.relu(total + self.addition)
        elif self.a == 'sigmoid':
            total = 0

            for i in range(len(inputs)):
                total += inputs[i] * self.multiplier[i]

            return MathENG1.sigmoid(total + self.addition)
        else:
            print("SYNTEX EROR")
            return
        
    def train(self, inputs, output, learning_rate):
        if len(self.multiplier) != len(inputs):
            print("INPUTS ERROR")
            return

        output_now = self.forward(inputs)

        error = output - output_now

        for i in range(len(inputs)):
            self.multiplier[i] += error * inputs[i] * learning_rate

        self.addition += error * learning_rate
        
     
#layer
class Layer:
    def __init__(self, input_size, neuron_count,activation='gelu'):
        self.si=input_size
        self.nc=neuron_count

        self.neurons=[]
        for i in range(neuron_count):
            self.neurons.append(Neuron(input_size,activation))

    def forward(self,inputs):
        answer = []
        
        for neurons in self.neurons:
            answer.append(neurons.forward(inputs))

        return answer
    
    def train(self, inputs, output, learning_rate):
        for i in range(len(self.neurons)):
            self.neurons[i].train(
                inputs,
                output[i],
                learning_rate
            )

#network
class Network:
    def __init__(self, neoron_inputs, layers, neoron_on_layers, output_neoron, activation='gelu'):
        self.ni = Layer(neoron_inputs, neoron_on_layers, activation)

        self.layers=[]
        self.layers.append(Layer(neoron_on_layers, neoron_on_layers, activation))

        for i in range(layers - 1):
            self.layers.append(
                Layer(neoron_on_layers, neoron_on_layers, activation)
            )

        self.on = Layer(neoron_on_layers, output_neoron, activation)

    def forward(self,inputs):
        output = self.ni.forward(inputs)

        for layer in self.layers:
            output = layer.forward(output)

        output = self.on.forward(output)

        return output
    
    def train(self, inputs, output, learning_rate):
        data = self.ni.forward(inputs)

        layer_outputs = []

        for layer in self.layers:
            data = layer.forward(data)
            layer_outputs.append(data)

        final_output = self.on.forward(data)

        self.on.train(
            data,
            output,
            learning_rate
        )
