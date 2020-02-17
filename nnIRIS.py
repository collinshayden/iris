#libraries
import pandas as pd
import numpy as np
import os
import random

#importing data as dataArray
path = "/Users/haydencollins/Desktop/Comp-Sci/data/irisdata.csv"
dataArray = np.genfromtxt(path,delimiter=",",dtype=None,encoding='utf-8-sig')

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def sigmoid_derivative(x):
    return (sigmoid(x)*(1-sigmoid(x)))

class Iris(object):
    def __init__(self,dataArray):
        self.dataArray = dataArray

        #empty arrays
        self.inputData = []
        self.strTargets = []
        self.targets = np.zeros((150,1))

        self.trainData = []
        self.trainTargets = []

        self.testData = []
        self.testTargets = []

    def Data(self):#organizes data
        for i in self.dataArray:
            self.inputData.append([i[0],i[1],i[2],i[3]])#creates inputData array with only features
            self.strTargets.append(i[4])#creates targets array with only classifications
        
        #turning the array of targets as strings into integer values so the nn can read them
        for i in range(len(self.strTargets)):
            if(self.strTargets[i] == 'Iris-setosa'):
                self.targets[i] = 0
            elif(self.strTargets[i] == 'Iris-versicolor'):
                self.targets[i] = 1
            elif(self.strTargets[i] == 'Iris-virginica'):
                self.targets[i] = 2

        self.trainData = np.concatenate((self.inputData[0:40],self.inputData[50:90],self.inputData[100:140]))#80% of data
        self.trainTargets = np.concatenate((self.targets[0:40],self.targets[50:90],self.targets[100:140]))#corresponding 80% of classifications

        self.testData = np.concatenate((self.inputData[40:50],self.inputData[90:100],self.inputData[140:150]))#20% of data
        self.testTargets = np.concatenate((self.targets[40:50],self.targets[90:100],self.targets[140:150]))#corresponding 20% of classifications

class NeuralNetwork:
    def __init__(self,x,y):
        self.input = np.asarray(x)
        self.weights1 = np.random.rand(self.input.shape[1],5)#input to hidden
        self.weights2 = np.random.rand(5,1)#hidden to output
        self.y = np.asarray(y)
        self.output = np.zeros(self.y.shape)#fills an array with the size of y with zeros


    def Feedforward(self):#multiplies layers together 
        self.layer1 = sigmoid(np.dot(self.input,self.weights1))
        self.output = sigmoid(np.dot(self.layer1,self.weights2))
    
    def sum_of_squares_error(self):#gives a numerical value to represent the absolute value of error
        sum = 0
        for i in range(0,len(self.y)):
            sum += (self.y[i]-self.output[i])**2#** applies an exponent
        self.loss = sum

        self.loss_derivative = (2 * (self.y - self.output[0]) * sigmoid_derivative(self.output[0]))

    def BackPropagation(self):
        #goal of backpropagation is to go backwards through the neuralnet and make small changes to weights(and bias' if present)...
        #to make the difference between target output and predicted output as close to zero as possible(which is the loss function)
        

        #dot product of first layer and derivative of loss function
        self.d_weights2 = np.dot(self.layer1.T, self.loss_derivative)
        #dot product of input and dot product of derivative of loss function with weights*derivative of first layer
        self.d_weights1 = np.dot(self.input.T, (np.dot(self.loss_derivative, self.weights2.T) * sigmoid_derivative(self.layer1)))

        #updating weights in accordance to the loss function
        self.weights1 += self.d_weights1
        self.weights2 += self.d_weights2


iris = Iris(dataArray)#creates an Iris class with dataArray(includes both data and targets) as the input

#calling iris methods
iris.Data()


nn = NeuralNetwork(iris.inputData,iris.targets)
nn.Feedforward()
nn.sum_of_squares_error()
nn.BackPropagation()

print(nn.output)
