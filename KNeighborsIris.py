#libraries
import numpy as np
import os
import random
from sklearn.neighbors import KNeighborsClassifier

#importing data as dataArray
path = "/Users/haydencollins/Desktop/Comp-Sci/misc/irisdata.csv"
dataArray = np.genfromtxt(path,delimiter=",",dtype=None,encoding='utf-8-sig')

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

        self.sepal_length = []
        self.sepal_width = []
        self.petal_length = []
        self.petal_width = []

        self.KNC = KNeighborsClassifier(n_neighbors=5)

    def Data(self):#organizes data
        for i in self.dataArray:
            self.inputData.append([i[0],i[1],i[2],i[3]])#creates inputData array with only features
            self.targets.append(i[4])#creates targets array with only classifications

        self.trainData = np.concatenate((self.inputData[0:40],self.inputData[50:90],self.inputData[100:140]))#80% of data
        self.trainTargets = np.concatenate((self.targets[0:40],self.targets[50:90],self.targets[100:140]))#corresponding 80% of classifications

        self.testData = np.concatenate((self.inputData[40:50],self.inputData[90:100],self.inputData[140:150]))#20% of data
        self.testTargets = np.concatenate((self.targets[40:50],self.targets[90:100],self.targets[140:150]))#corresponding 20% of classifications
        

    def Train(self):
        #KNC.fit(X,y) where X is training data and y is targets. This function is training the algorithm to 80% of the dataset.
        self.KNC.fit(self.trainData,self.trainTargets)

    def Predict(self):
        #KNC.predict(X) where X is unlabelled data, takes the data points and compares them to nearby labelled data to identify a classification
        self.prediction = self.KNC.predict(self.testData)
        print(self.prediction)

    def Scoring(self):
        self.score=0
        for i in range(0,len(self.prediction),1):
            if(self.prediction[i] == self.testTargets[i]):#if the predictions and targets are the same:
                self.score += 1#increase score by 1

        #printing values
        print("Predictions Correct = "+str(self.score))
        print("Predictions Made = "+str(len(self.prediction)))

        self.Accuracy = (self.score/len(self.testData))*100
        self.percentAccuracy=(str(self.Accuracy)+"%")

        print("Percent Accuracy = "+str(self.percentAccuracy))

iris = Iris(dataArray)#creates an Iris class with dataArray(includes both data and targets) as the input

#calling iris methods
Iris.Data(iris)
Iris.Train(iris)
Iris.Predict(iris)
Iris.Scoring(iris)
