#libraries
import pandas as pd
import numpy as np
import os
import random
from sklearn.neighbors import KNeighborsClassifier

#importing data as dataArray
path = "/Users/haydencollins/Desktop/Comp-Sci/misc/irisdata.csv"
dataArray = np.genfromtxt(path,delimiter=",",dtype=None,encoding='utf-8-sig')

class Iris(object):
    def __init__(self,input_nodes,output_nodes,dataArray):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.dataArray = dataArray
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
        self.trainData.append([self.inputData[0:40],self.inputData[50:90],self.inputData[100:140]])#80% of data
        self.trainTargets.append([self.targets[0:40],self.targets[50:90],self.targets[100:140]])#corresponding 80% of classifications

        self.testData.append([self.inputData[40:50],self.inputData[90:100],self.inputData[140:150]])#20% of data
        self.testTargets.append([self.targets[40:50],self.targets[90:100],self.targets[140:150]])#corresponding 20% of classifications


iris = Iris(4,3,dataArray)
Iris.Data(iris)




