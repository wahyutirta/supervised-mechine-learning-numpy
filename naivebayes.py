# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 12:15:37 2021

@author: User
"""

import numpy as np

import matplotlib.pyplot as plt

from data import *

class NaiveBayes:
    
    def fit (self, x, y):
        n_samples, n_features = x.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for c in self._classes:
            x_c = x[c==y]
            self._mean[c,:] = x_c.mean(axis= 0)
            self._var[c,:] = x_c.var(axis= 0)
            self._priors[c] = x_c.shape[0] / float(n_samples)
    
    def predict(self, x):
        """
        predict input sample
        
        return sample prediction for each sample
        """
        y_pred = [self._predict(sample) for sample in x]
        return y_pred
    
    def _predict(self, x):
        """
        predict single sample
        """
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._probDensityFunc(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        
        return self._classes[np.argmax(posteriors)]
            
    def _probDensityFunc(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(- np.power((x-mean),2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def calculateACC(self, predictions, y_test):
        if len(y_test.shape) == 2:
            y_test = np.argmax(y, axis=1)
            # np.argmax return indexs of max value each row (axis 1)
            # np.argmax return array of index refering to position of maximun value along axis 1
        accuracy = np.mean(predictions==y_test)
        print(f'Naive Bayes predict acc :: {accuracy:.3f}')

x, y = vertical.create_data(samples=1000, classes=5)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='brg') 
plt.show()

x_test, y_test = vertical.create_data(samples=100, classes=5)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap='brg') 
plt.show()


nb = NaiveBayes()
nb.fit(x, y)
predictions = nb.predict(x_test)

nb.calculateACC(predictions, y_test)