# from classes.Model import Model

import collections
import pickle

import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from scipy.stats import norm, multivariate_normal

np.random.seed(2)
x1 = np.random.normal(0, 2, size = 2000)
x2 = np.random.normal(5, 5, size = 2000)
data = [x1, x2]

def plot_hist(data):
    for x in data:
        plt.hist(x, bins = 80, normed =True, alpha = 0.6)

    plt.xlim(-10, 20)
    plt.show()

plot_hist(data)

# class HMM():
#     def __init__(self):
#         # Init Properties
#         self.validation_prop = validation_prop #Percentage of the data used for training
#         self.cepest_coeif = cepest_coeif #Features used in speech recognition

#     #def train(self):
