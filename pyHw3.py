#Frederic Boileau 
#HW3 IFT6269-A2018
import numpy as np
import numpy.linalg as lina
import pandas as pd
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
from copy import deepcopy
from operator import attrgetter

import os
import errno
from pdb import set_trace as bp
import json

def script():
  dataTest = pd.DataFrame(pd.read_csv("hwk3data/EMGaussian.test", sep=' '))
  dataTrain = pd.DataFrame(pd.read_csv("hwk3data/EMGaussian.train", sep=' '))
  nbRestarts = 10
  k = 4

  e = EM(True, dataTest, dataTrain)
  e._seed()
  return e

# Centroids: self.train.sample(n=4).values
# Labels : np.zeros(len(self.train))
class K_Means_Results():
  def __init__(self, centroids, labels, trainData, testData, kmeansScore=None):
    self.centroids = centroids
    self.labels = labels
    self.testData = testData
    self.trainData = trainData
    self.kmeansScore = (kmeansScore if kmeansScore is not None
                        else self._kMeansScore())

  def _kMeansScore(self):
    score = 0
    for value in self.testData.values:
      distances = []
      for c in self.centroids:
        distances.append(dist(value,c,ax=0)**2)
      score += np.min(distances)
    return score

  def save(self, file):
    data = {'Score' : self.kmeansScore, 'centroids' : self.centroids.tolist()}
    json.dump(data, file, indent=2)

class Kmeans():
  def __init__(self, test, train, K=4, nbRestarts=10):
    self.K = K
    self.test = test
    self.train = train
    self.results = self._randomRestarts(nbRestarts)
    self.best = min(self.results, key=attrgetter('kmeansScore'))

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
    labels = np.zeros(len(self.train))
    values = self.train.values
    epsilon = dist(C, C_old, None)

    while epsilon != 0.0:
      #Assign points to closest centroid through an ordred list of labels
      for i in range(len(self.train)):
        distances = dist(values[i], C)
        closest = np.argmin(distances)
        labels[i] = closest

      C_old = deepcopy(C)
      #The new centroid is the average of the points closest to the old one
      for i in range(self.K):
        points = [values[j] for j in range(len(values)) if labels[j]==i]
        C[i] = np.mean(points, axis=0)

      epsilon = dist(C, C_old, None)

    return K_Means_Results(centroids=C, labels=labels,
                           trainData=self.train, testData=self.test)

  def plot(self, saveOrDisplay, directory=None):
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    values = self.train.values
    for figId, result in enumerate(self.results):
      fig = plt.figure(figId)
      ax = fig.add_subplot(111)
      C = result.centroids

      for i in range(self.K):
        points = np.array([values[j] for j in \
                           range(len(values)) if result.labels[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
        ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')

      if (saveOrDisplay == "save"):
        dir = "./figs" if directory is None else directory
        fig.savefig(dir + '/K_means_fig' + str(figId))

    if (saveOrDisplay == "display"):
      plt.show()

  def saveResults(self, directory):
    createFolder(directory)
    filename = directory + "/K_Means"
    with open(filename, "w") as file:
      for result in self.results:
        result.save(file)


class EM():
  def __init__(self, IsoOrNot : bool, test, train, K=4, nbRestarts=10):
    self.IsoOrNot = IsoOrNot
    self.test = test
    self.train = train
    self.K = K
    self.N = len(self.train)
    self.upToN = range(len(self.train))
    self.KMeansResults = Kmeans(test,train, K, nbRestarts).best

  def _seed(self):
    Mu = self.KMeansResults.centroids
    groups = EM._groupByLabel(self.train.values, self.KMeansResults.labels, self.K)
    Sigma = [map(np.std, group) for group in groups]
    Nks = [len(group) for group in groups]
    Pi = [Nk/self.N for Nk in Nks]

    return dict(Mu=Mu, Sigma=Sigma, Nks=Nks, Pi=Pi)

  def _responsabilities(self, X, Pi, Mu, Sigma):
    Normal = [multivariate_normal(mean=mu, cov=sig).pdf
              for mu,sig in zip(Mu,Sigma)]

    Gamma_ = [[pi*normal(x) for pi,normal in zip(Pi,Normal)] for x in X]
    Gamma = [[gamma/sum(row) for gamma in row] for row in _Gamma]
    return np.array(Gamma)

  def _piNew(self, Nks):
    return [Nk/self.N for Nk in Nks]

  def _muNew(self, X, Nk, Gamma, Resp):
   # muNew_ = list(np.einsum('ij,ij->j',Gamma, X))
   # muNew = [mu_/Nk for (mu_,Nk) in zip(muNew_,Nk)]

  def sigNewIso(self, X, Nk, Mu, Resp):
    deviations = [[(xn - muk)**2 for muk in Mu] for xn in 
  def sigNewGen(self, X, Nk, Mu, Resp):
    pass

  @staticmethod
  def _groupByLabel(values, labels, K):
    upToN = range(len(values))
    groups = [[values[j] for j in upToN if labels[j] == i] for i in range(K)]
    return groups



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

