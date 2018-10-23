#Frederic Boileau 
#HW3 IFT6269-A2018
import copy
import numpy as np
import numpy.linalg as lina
import pandas as pd
import matplotlib.pyplot as plt
from math import log
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import os
import errno

from pdb import set_trace as bp

pathForFigures = "../texhwk3/figures/"
directory = "./dump/"

def script():
  interactive = True

  if interactive:
    pathForFigures = createFolder(directory)
  dataTest = pd.DataFrame(pd.read_csv("hwk3data/EMGaussian.test", sep=' '))
  dataTrain = pd.DataFrame(pd.read_csv("hwk3data/EMGaussian.train", sep=' '))

  K = Kmeans(dataTest, dataTrain)
  K.clusterize()


  return dataTest, dataTrain


class Kmeans():
  def __init__(self, test, train, K=4):

    self.K = K
    self.test = test
    self.train = train

    self.listOfLabelsList = list()
    self.lifstOfC = list()

  def _clusterize(self):
    C = self.train.sample(n=4).values
    C_old = np.zeros(C.shape)
    labelsList = np.zeros(len(self.train))
    values = self.train.values
    epsilon = self._dist(C, C_old, None)

    while epsilon != 0.0:
      #Assign points to closest centroid through an ordred list of labels
      for i in range(len(self.train)):
        distances = self._dist(values[i], C)
        closest = np.argmin(distances)
        self.labelsList[i] = closest

      C_old = copy.deepcopy(C)
      #The new centroid is the average of the points closest to the old one
      for i in range(self.K):
        points = [values[j] for j in range(len(values)) if labelsList[j]==i]
        C[i] = np.mean(points, axis=0)

      epsilon = self._dist(C, C_old, None)
    return C, labelsList

  def _dist(self, a, b, ax=1):
    return lina.norm(a-b, axis=ax)

  def plot(self):
    pass

def createFolder(directory):
  try:
      if not os.path.exists(directory):
          os.makedirs(directory)
      return directory
  except OSError:
      print ('Error: Creating directory. ' +  directory)

if __name__ == '__main__':
  script()
