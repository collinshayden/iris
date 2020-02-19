#libraries
import numpy as np
import random
import math

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def sigmoid_derivative(x):
    return ((x)*(1-(x)))
#note: this is not the derivative of the sigmoid function
#because the sigmoid function is already applied to nn.layer1 and nn.output
#and therefore the x input of this function has already had the sigmoid function applied to it
#so this function does not need to apply the sigmoid function to x

simpleArrayInputs = np.asarray([[1,1,1],[1,0,1],[1,1,0],[1,1,1]])
simpleArrayTargets = np.asarray([[0],[1],[1],[0]])

class NeuralNetwork:
    def __init__(self,x,y):
        self.input = np.asarray(x)
        self.weights1 = np.random.rand(self.input.shape[1],4)#input to hidden
        self.weights2 = np.random.rand(4,1)#hidden to output
        self.y = np.asarray(y)
        self.output = np.zeros(self.y.shape)#fills an array with the size of y with zeros

    def Feedforward(self):#multiplies layers together 
        self.layer1 = sigmoid(np.dot(self.input,self.weights1))
        self.output = sigmoid(np.dot(self.layer1,self.weights2))
    
    def sum_of_squares_error(self):#gives a numerical value to represent the absolute value of error
        self.loss = ((self.output[0] - self.y)**2).sum()#note: ** is the python exponent operator
    
    def BackPropagation(self):
        #goal of backpropagation is to go backwards through the neuralnet and make small changes to weights(and bias' if present)...
        #to make the difference between target output and predicted output as close to zero as possible by using the loss function
        
        #chain rule derivative of the loss function
        self.loss_derivative = (2 * (self.y - self.output[0]) * sigmoid_derivative(self.output[0]))
        #layer1 * loss derivative
        self.d_weights2 = np.dot(self.layer1.T, self.loss_derivative)
        #input * ((loss derivative * weights2) * derivative of layer 1)
        self.d_weights1 = np.dot(self.input.T, (np.dot(self.loss_derivative, self.weights2.T) * sigmoid_derivative(self.layer1)))

        #updating weights for next training cycle
        self.weights1 += self.d_weights1
        self.weights2 += self.d_weights2

    def runTrainingCycle(self):#iterates to apply changes made in backpropagation
        self.Feedforward()
        self.BackPropagation()
        self.sum_of_squares_error()

nn = NeuralNetwork(simpleArrayInputs,simpleArrayTargets)#takes input data and target values 

for i in range(1500):
    nn.runTrainingCycle()

print("nn.input")
print(nn.input)
print("nn.weights1")
print(nn.weights1)
print("nn.layer1")
print(nn.layer1)
print("nn.weights2")
print(nn.weights2)
print("nn.output")
print(nn.output)

roundedOutput = []
for i in range(len(nn.output)):
    roundedOutput.append(round(nn.output[i][0],0))

print("roundedOutput")
print(roundedOutput)
print("targetOutput")
print(nn.y.T)
