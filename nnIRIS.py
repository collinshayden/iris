#libraries
import pandas as pd
import numpy as np
import os
import random

#importing data as dataArray
path = "/Users/haydencollins/Desktop/Comp-Sci/misc/irisdata.csv"
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
        self.targets = []

        self.trainData = []
        self.trainTargets = []

        self.testData = []
        self.testTargets = []

    def Data(self):#organizes data
        for i in self.dataArray:
            self.inputData.append([i[0],i[1],i[2],i[3]])#creates inputData array with only features
            self.targets.append(i[4])#creates targets array with only classifications

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
    
    def Feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input,self.weights1))
        self.output = sigmoid(np.dot(self.layer1,self.weights2))
    
    def BackPropagation(self):
        #using chain rule to find derivative of loss function with weights 
        self.d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        self.d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        #updating weights in accordance to the loss function
        self.weights1 += self.d_weights1
        self.weights2 += self.d_weights2

iris = Iris(dataArray)#creates an Iris class with dataArray(includes both data and targets) as the input

#calling iris methods
Iris.Data(iris)


nn = NeuralNetwork(iris.inputData,iris.targets)
NeuralNetwork.Feedforward(nn)
NeuralNetwork.BackPropagation(nn)




