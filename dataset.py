import pandas as pd
import numpy as np
import pickle as p
import random

class dataset:
    def __init__(self, filename='./data/iris.csv', percentLabeled=0.25, numLabeled=0, header=None):
        self.masterData = pd.read_csv(filename, header)
        self.masterShape = self.masterData.shape

        if numLabeled > 0:
            labeled_shape = (size, self.shape[1])
        else:
            labeled_shape = (int(self.masterShape[0]*percentLabeled), self.masterShape[1])

        unlabeledInd = random.sample(range(self.masterShape[0]), labeled_shape[0])
        print(self.masterShape)
        print(self.masterData)
        print(unlabeledInd)
        # self.unlabeled = self.masterData[unlabeledInd]
