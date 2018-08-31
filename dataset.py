import pandas as pd
import numpy as np
import pickle as p
import random

class Dataset:
    def __init__(self, filename='./data/iris.csv', features=['sepal length', 'sepal width', 'petal length', 'petal width', 'label']):
        self.masterData = pd.read_csv(filename, names=features)
        self.features = features
        self.masterShape = self.masterData.shape
        self.labels = self.masterData[features[-1]].unique()

    def createRandomSampling(self, percentLabeled=0.5, initLabeling=True, numLabeled=0,):
        if numLabeled > 0:
            labeled_shape = (size, self.shape[1])
        else:
            labeled_shape = (int(self.masterShape[0]*percentLabeled), self.masterShape[1])

        isLabeled = [False] * self.masterShape[0]
        for i in random.sample(range(self.masterShape[0]), labeled_shape[0]):
            isLabeled[i] = True

        self.labels = self.masterData[self.features[-1]].unique()
        self.labeledData = {l : {feat : [] for feat in self.features[:-1]} for l in self.labels}

        self.unlabeledData = {feat : [] for feat in self.features[:-1]}

        # Adding the labeled / unlabeled data
        for ind in range(self.masterShape[0]):
            if isLabeled[ind]:
                label = self.masterData['label'][ind]
                for feat in self.features[:-1]:
                    self.labeledData[label][feat].append(self.masterData[feat][ind])
            else:
                for feat in self.features[:-1]:
                    self.unlabeledData[feat].append(self.masterData[feat][ind])

        self.selectedData = {feat: [] for feat in self.features[:-1]}

        return {'labeled': self.labeledData, 'unlabeled': self.unlabeledData, 'selected': self.selectedData}

    def getLabeledData(self):
        return self.labeledData

    def getUnlabeledData(self):
        return self.unlabeledData

    def getEmptySelectedData(self):
        return self.selectedData

    def labelData(self, payload):
        unlabeled = payload['unlabeled']
        labeled = payload['labeled']
        selected= payload['selected']

        for selectedInd in range(len(selected[self.features[0]])):
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

        return {'labeled':labeled, 'unlabeled':unlabeled, 'selected':{feat: [] for feat in self.features[:-1]}}
