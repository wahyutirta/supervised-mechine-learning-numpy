# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 11:34:11 2021

@author: User
"""
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

from data import *

class KNN:
    def __init__(self, k=3):
        self.k = k
        
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum(np.power((x1 - x2), 2)))
    
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y
    
    def predict (self, x):
        y_pred = [self._predict(elemen) for elemen in x]
        return np.array(y_pred)
    
    def _predict(self, x):
        #compute sample's distances
        distances = [self.euclidean_distance(x_train, x) for x_train in self.x_train]
        
        #key indeces, sort k nearest samples, label
        k_idx = np.argsort(distances)[: self.k]
        #most common class label
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        #top (1) most common class
        most_common = Counter(k_neighbor_labels).most_common(1)
        
        #return most common result
        return most_common[0][0]
    
    def calculateACC(self, predictions, y_test):
        if len(y_test.shape) == 2:
            y_test = np.argmax(y, axis=1)
            # np.argmax return indexs of max value each row (axis 1)
            # np.argmax return array of index refering to position of maximun value along axis 1
        accuracy = np.mean(predictions==y_test)
        print(f'KNN predict acc :: {accuracy:.3f}')
    
x, y = vertical.create_data(samples=1000, classes=5)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='brg') 
plt.show()

x_test, y_test = vertical.create_data(samples=100, classes=5)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap='brg') 
plt.show()

model = KNN(k=5)
model.fit(x, y)
predictions = model.predict(x_test)

model.calculateACC(predictions, y_test)
    
        
        