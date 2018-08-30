import pandas as pd
import numpy as np
import pickle as p
import random

class Dataset:
    def __init__(self, filename='./data/iris.csv', percentLabeled=0.25, initLabeling=True, numLabeled=0, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'label']):
        self.masterData = pd.read_csv(filename, names=names)
        self.masterShape = self.masterData.shape
        self.labels = self.masterData[names[-1]].unique()

        if initLabeling:
            if numLabeled > 0:
                labeled_shape = (size, self.shape[1])
            else:
                labeled_shape = (int(self.masterShape[0]*percentLabeled), self.masterShape[1])

            labeledInd = random.sample(range(self.masterShape[0]), labeled_shape[0])
            self.labeledData = {l : {feat : [] for feat in names[:-1]} for l in self.labels}

            # Adding the labeled data
            for ind in labeledInd:
                label = self.masterData['label'][ind]
                for feat in names[:-1]:
                    self.labeledData[label][feat].append(self.masterData[feat][ind])

    def getLabeledData(self):
        return self.labeledData

