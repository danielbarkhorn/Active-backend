import pandas as pd
import numpy as np
import pickle as p
import random
from functools import reduce

class Dataset:
    def __init__(self, filename='./data/iris.csv', features=['sepal length', 'sepal width', 'petal length', 'petal width', 'label']):
        self.masterData = pd.read_csv(filename, names=features)
        self.features = features
        self.masterShape = self.masterData.shape
        self.labels = self.masterData[features[-1]].unique()
        self.encoding = {self.labels[i] : i for i in range(len(self.labels))}

    def createRandomSampling(self, percentLabeled=0.3, percentTest=0.3, initLabeling=True, numLabeled=0,):
        if numLabeled > 0:
            self.labeled_shape = (numLabeled, self.masterShape[1])
        else:
            self.labeled_shape = (int(self.masterShape[0]*(percentLabeled + percentTest)), self.masterShape[1])

        isLabeled = [False] * self.masterShape[0]
        isTest = [False] * self.masterShape[0]
        labeledInd = random.sample(range(self.masterShape[0]), self.labeled_shape[0])

        testBreak = int(percentTest * self.masterShape[0])
        testInd = labeledInd[0:testBreak]
        labeledInd = labeledInd[testBreak:]

        for i in labeledInd:
            isLabeled[i] = True
        for i in testInd:
            isTest[i] = True

        self.test_X = []
        self.test_Y = []
        self.labels = self.masterData[self.features[-1]].unique()
        self.labeledData = {l : {feat : [] for feat in self.features[:-1]} for l in self.labels}
        self.unlabeledData = {feat : [] for feat in self.features[:-1]}

        for ind in range(self.masterShape[0]):
            if isLabeled[ind]:
                label = self.masterData['label'][ind]
                for feat in self.features[:-1]:
                    self.labeledData[label][feat].append(self.masterData[feat][ind])
            elif isTest[ind]:
                self.test_X.append(list(self.masterData.iloc[ind, :-1]))
                self.test_Y.append(self.encoding[self.masterData.iloc[ind, -1]])
            else:
                for feat in self.features[:-1]:
                    self.unlabeledData[feat].append(self.masterData[feat][ind])

        labeledDF = self.masterData.iloc[labeledInd]
        self.labeled_X = labeledDF.iloc[:, :-1].values
        self.labeled_Y = [self.encoding[label] for label in labeledDF.iloc[:, -1]]

        self.selectedData = {feat: [] for feat in self.features[:-1]}

        return {'test_X': self.test_X, 'test_Y':self.test_Y, 'labeled': self.labeledData, 'unlabeled': self.unlabeledData, 'selected': self.selectedData}

    def get_X(self):
        return self.labeled_X

    def get_Y(self):
        return self.labeled_Y

    def getUnlabeled(self):
        return self.unlabeled

    def getLabeledData(self):
        return self.labeledData

    def getUnlabeledData(self):
        return self.unlabeledData

    def getEmptySelectedData(self):
        return self.selectedData

    def getTestData(self):
        return self.test

    def loadPayload(self, payload):
        unlabeled = payload['unlabeled']
        self.unlabeledDict = unlabeled
        labeled = payload['labeled']
        self.labeled = labeled
        selected = payload['selected']
        self.selected = selected

        self.unlabeled = np.zeros((len(unlabeled[self.features[0]]), len(self.features[:-1])))
        for unlabeledInd in range(len(unlabeled[self.features[0]])):
            self.unlabeled[unlabeledInd] = [unlabeled[feat][unlabeledInd] for feat in self.features[:-1]]

        numLabeled = sum([len(labeled[label][self.features[0]]) for label in self.labels])
        self.labeled_X = np.zeros((numLabeled, len(self.features[:-1])))
        self.labeled_Y = np.zeros((numLabeled, 1))
        labeledInd = 0
        for label in self.labels:
            for labeledInstanceInd in range(len(labeled[label][self.features[0]])):
                self.labeled_X[labeledInd] = [labeled[label][feat][labeledInstanceInd] for feat in self.features[:-1]]
                self.labeled_Y[labeledInd] = self.encoding[label]
                labeledInd += 1

    def labelData(self, payload, numLabeled):
        unlabeled = payload['unlabeled']
        labeled = payload['labeled']
        selected= payload['selected']

        for selectedInd in range(len(selected[self.features[0]])):
            numLabeled += 1
            selectedDP = {feat : selected[feat][selectedInd] for feat in self.features[:-1]}
            for unlabeledInd in range(len(unlabeled[self.features[0]])):
                unlabeledDP = {feat : unlabeled[feat][unlabeledInd] for feat in self.features[:-1]}
                if(selectedDP == unlabeledDP):
                    for feat in self.features[:-1]:
                        unlabeled[feat].pop(unlabeledInd)
                    break
            for masterInd in range(self.masterShape[0]):
                masterDP = {feat : self.masterData[feat][masterInd] for feat in self.features[:-1]}
                if(selectedDP == masterDP):
                    newClass = labeled[self.masterData[self.features[-1]][masterInd]]
                    for feat in self.features[:-1]:
                        newClass[feat].append(selectedDP[feat])
                    break

        return {'numLabeled':numLabeled, 'labeled':labeled, 'unlabeled':unlabeled, 'selected':{feat: [] for feat in self.features[:-1]}}
