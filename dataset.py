import pandas as pd
import numpy as np
import pickle as p
import random

class Dataset:
    def __init__(self, filename='./data/iris.csv', percentLabeled=0.5, initLabeling=True, numLabeled=0, features=['sepal length', 'sepal width', 'petal length', 'petal width', 'label']):
        self.masterData = pd.read_csv(filename, names=features)
        self.features = features
        self.masterShape = self.masterData.shape
        self.labels = self.masterData[features[-1]].unique()

        if initLabeling:
            if numLabeled > 0:
                labeled_shape = (size, self.shape[1])
            else:
                labeled_shape = (int(self.masterShape[0]*percentLabeled), self.masterShape[1])

            isLabeled = [False] * self.masterShape[0]
            for i in random.sample(range(self.masterShape[0]), labeled_shape[0]):
                isLabeled[i] = True

            self.labels = self.masterData[features[-1]].unique()
            self.labeledData = {l : {feat : [] for feat in features[:-1]} for l in self.labels}

            self.unlabeledData = {feat : [] for feat in features[:-1]}

            # Adding the labeled / unlabeled data
            for ind in range(self.masterShape[0]):
                if isLabeled[ind]:
                    label = self.masterData['label'][ind]
                    for feat in features[:-1]:
                        self.labeledData[label][feat].append(self.masterData[feat][ind])
                else:
                    for feat in features[:-1]:
                        self.unlabeledData[feat].append(self.masterData[feat][ind])

    def getLabeledData(self):
        return self.labeledData

    def getUnlabeledData(self):
        return self.unlabeledData

    def getEmptySelectedData(self):
        return {feat: [] for feat in self.features[:-1]}
