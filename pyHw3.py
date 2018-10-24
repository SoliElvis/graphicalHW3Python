#Frederic Boileau 
#HW3 IFT6269-A2018
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
from collections import namedtuple
from copy import deepcopy

pathForFigures = "../texhwk3/figures/"
directory = "./dump/"

#TODO : 1) Plot best with test data

def script():
  interactive = True

  if interactive:
    pathForFigures = createFolder(directory)
  dataTest = pd.DataFrame(pd.read_csv("hwk3data/EMGaussian.test", sep=' '))
  dataTrain = pd.DataFrame(pd.read_csv("hwk3data/EMGaussian.train", sep=' '))
  nbRestarts = 3
  k = 4
  K = Kmeans(dataTest, dataTrain, k, nbRestarts)
  K.plot()
  return K


# Centroids: self.train.sample(n=4).values
# Labels : np.zeros(len(self.train))
class K_Means_Results():
  def __init__(self, centroids, labels, testData, kmeansScore=None):
    self.centroids = centroids
    self.labels = labels
    self.testData = testData
    self.kmeansScore = (kmeansScore if kmeansScore is not None
                        else self._kMeansScore())

  def _kMeansScore(self):
    score = 0
    for value in self.testData.values:
      distances = []
      for c in self.centroids:
        distances.append(dist(value,c,ax=0))
      score += np.min(distances)
    return score

class Kmeans():

  def __init__(self, test, train, K=4, nbRestarts=10):
    self.K = K
    self.test = test
    self.train = train
    self.results = self._randomRestarts(nbRestarts)

  def _randomRestarts(self, nbRestarts):
    best = self._clusterize()
    clusterResults = [best]
    for i in  range(nbRestarts):
      newRestart = self._clusterize()
      if (newRestart not in clusterResults):
        if (newRestart.kmeansScore > best.kmeansScore):
          best = deepcopy(newRestart)

        result = deepcopy(newRestart)
        clusterResults.append(result)

    return clusterResults

  #Creates new K_Means_Results object with random seed of centroids
  def _clusterize(self):
    C = self.train.sample(n=4).values
    C_old = np.zeros(C.shape)
    labelsList = np.zeros(len(self.train))
    values = self.train.values
    epsilon = dist(C, C_old, None)

    while epsilon != 0.0:
      #Assign points to closest centroid through an ordred list of labels
      for i in range(len(self.train)):
        distances = dist(values[i], C)
        closest = np.argmin(distances)
        labelsList[i] = closest

      C_old = deepcopy(C)
      #The new centroid is the average of the points closest to the old one
      for i in range(self.K):
        points = [values[j] for j in range(len(values)) if labelsList[j]==i]
        C[i] = np.mean(points, axis=0)

      epsilon = dist(C, C_old, None)
    return K_Means_Results(centroids=C, labels=labelsList, testData=self.test)

  def plot(self, saveOrDisplay):
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    values = self.train.values
    for result in self.results:
      fig, ax = plt.subplots()
      C = result.centroids
      for i in range(self.K):
        points = np.array([values[j] for j in \
                           range(len(values)) if result.labels[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
        ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')

    if (saveOrDisplay == "display"):
      plt.show()

    if (saveOrDisplay == "save"):
      pass

#Small helper functions
def dist(a, b, ax=1):
  return lina.norm(a-b, axis=ax)

def createFolder(directory):
  try:
      if not os.path.exists(directory):
          os.makedirs(directory)
      return directory
  except OSError:
      print ('Error: Creating directory. ' +  directory)

if __name__ == '__main__':
  script()


#    def quickTest():
#      colors = ['r', 'g', 'b', 'y', 'c', 'm']
#      fig, ax = plt.subplots()
#      for i in range(self.K):
#        points = np.array([values[j] for j in \
#                           range(len(values)) if labelsList[j] == i])
#        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
#        ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
#      plt.show()
#
#
