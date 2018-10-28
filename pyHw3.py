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
from IPython.core import debugger
import json

Idebug = debugger.Pdb().set_trace

def script():
  dataTest = pd.DataFrame(pd.read_csv("hwk3data/EMGaussian.test", sep=' '))
  dataTrain = pd.DataFrame(pd.read_csv("hwk3data/EMGaussian.train", sep=' '))
  nbRestarts = 10
  K = 4
  nbIter = 20
  interactive = False
  k = Kmeans(dataTest, dataTrain,K,nbRestarts)
  k.plot(interactive)
  e = EM(dataTest, dataTrain,k.best,K,nbRestarts,nbIter)
  e.runAndPlotBoth(interactive)

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

  def plot(self, interactive=True, directory="./figs"):
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    values = self.train.values
    for figId, result in enumerate(self.results):
      fig = plt.figure(figId+10)
      ax = fig.add_subplot(111)
      C = result.centroids

      for i in range(self.K):
        points = np.array([values[j] for j in \
                           range(len(values)) if result.labels[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
        ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')

      if (not interactive):
        fig.savefig(directory+ '/K_means_fig' + str(figId),format='svg')

    if (interactive):
      plt.show()

  def saveResults(self, directory):
    createFolder(directory)
    filename = directory + "/K_Means"
    with open(filename, "w") as file:
      for result in self.results:
        result.save(file)

class Eupdates():
  def __init__(self, Gamma, Nks):
    self.gamma = Gamma #Nxk
    self.Nks = Nks
class Mupdates():
  def __init__(self, Pi, Mu, Sigma):
    self.Pi = Pi
    self.Mu = Mu
    self.Sigma = Sigma
class EM():
  def __init__(self, test, train, KmeansResults=None, K=4, nbRestarts=3, nbIter=10):
    self.test = test
    self.train = train
    self.D = 2
    self.X = self.train.values
    self.K = K
    self.N = len(self.train)
    self.upToN = range(len(self.train))

    self.KMeansResults = (Kmeans(test,train, K, nbRestarts).best if KmeansResults is None
                         else KmeansResults)
    self.nbIter = nbIter

    self.mUpdate1 = None
    self.mUpdate2 = None

  def runAndPlotBoth(self, interactive):
    mUpdate1 = self.run(True) if self.mUpdate1 is None else self.mUpdate1
    mUpdate2 = self.run(False) if self.mUpdate2 is None else self.mUpdate2
    self.plot(mUpdate1, 1, interactive)
    self.plot(mUpdate2, 2, interactive)
    if (interactive):
      plt.show()

  def run(self, propToId):
    mUpdate = self._seed()
    for i in range(self.nbIter):
      eUpdate = self._eUpdate(mUpdate)
      mUpdate = self._mUpdate(eUpdate, propToId)
    return mUpdate

  def plot(self, mUpdate, figId, interactive=True, directory="./figs"):
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    values = self.test.values
    labels = self._label(mUpdate, values)
    fig = plt.figure(figId)
    ax = fig.add_subplot(111)

    for k in range(self.K):
      pi = mUpdate.Pi[k]
      mu = mUpdate.Mu[:,k]
      sig = mUpdate.Sigma[:,:,k]
      fun = multivariate_normal(mean=mu,cov=sig).pdf

      EM._plotContours(fun, figId,colors[k], pi)
      points = np.array([values[j] for j in range(len(values)) if labels[j] == k])
      ax.scatter(points[:, 0], points[:, 1],alpha=0.5, s=7, c=colors[k])
      ax.scatter(mu[0], mu[1], marker='+', s=100, c='#050505')

    if (interactive):
      plt.draw()
    else:
      fig.savefig(directory + '/EM_fig' + str(figId),format='svg')

  def _label(self, mUpdate, values):
    labels = np.zeros(len(self.test))
    funs = []
    for k in range(self.K):
      funs.append(multivariate_normal(mean=mUpdate.Mu[:,k],cov=mUpdate.Sigma[:,:,k]).pdf)

    for i in range(len(values)):
      pdfValues = [fun(values[i,:]) for fun in funs]
      best = np.argmax(pdfValues)
      labels[i] = best

    return labels

  def _seed(self):
    groups = EM._groupByLabel(self.X, self.KMeansResults.labels, self.K)
    Mu = self.KMeansResults.centroids.T
    Sigma = np.swapaxes(np.array([np.std(group)*np.identity(2) for group in groups]),0,2)
    Sigma_bad = np.swapaxes(np.array([5*np.identity(2) for i in range(4)]),0,2)
    Nks = [len(group) for group in groups]
    Pi = [Nk/self.N for Nk in Nks]
    return Mupdates(Pi, Mu, Sigma)

  def _eUpdate(self, Mupdates):
    mUpdate = deepcopy(Mupdates)
    Pi = mUpdate.Pi
    Mu = mUpdate.Mu
    Sigma = mUpdate.Sigma

    Normal = list()
    for k in range(self.K):
      pdf = multivariate_normal(mean=Mu[:,k], cov=Sigma[:,:,k]).pdf
      Normal.append(pdf)

    Gamma_ = np.empty([self.N, self.K])
    for k in range(self.K):
      normal = Normal[k]
      for i in range(self.N):
        Gamma_[i,k] = Pi[k]*normal(self.X[i,:])

    for i in range(self.N):
      rowSum = np.sum(Gamma_[i,:])
      for k in range(self.K):
        Gamma_[i,k] = Gamma_[i,k]/rowSum

    Nks = np.empty(self.K)
    for k in range(self.K):
      Nks[k] = np.sum(Gamma_[:,k])
    return Eupdates(Gamma_, Nks)

  def _mUpdate(self, eUpdate, propToId):
    PiNew = self._piNew(eUpdate.Nks)
    MuNew = self._muNew(eUpdate)
    SigNew = None
    if (propToId == True):
      SigNew = self._sigNewIso(eUpdate, MuNew)
    elif (propToId == False):
      SigNew = self._sigNewGen(eUpdate, MuNew)

    return Mupdates(PiNew, MuNew, SigNew)

  def _piNew(self, Nks):
    return [Nk/self.N for Nk in Nks]

  def _muNew(self, Eupdates):
    eUpdate = deepcopy(Eupdates)
    Gamma = eUpdate.gamma
    Nks = eUpdate.Nks
    muNew_ = np.empty([self.N,self.D,self.K])
    muNew = np.empty([self.D, self.K])
    for i in range(self.N):
      for k in range(self.K):
        muNew_[i,:,k] = self.X[i,:]*Gamma[i,k]

    for k in range(self.K):
      muNew[:,k] = np.sum(muNew_[:,:,k],axis=0)/Nks[k]

    return muNew

  def _sigNewIso(self, eUpdates, Mu):
    Nks = eUpdates.Nks
    sigma = np.empty([self.D, self.D, self.K])#DxDxK
    deviations = np.empty([self.N, self.K])#NDxK
    wDeviations = np.empty([self.N, self.K])
    Gamma = eUpdates.gamma

    for k in range(self.K):
      for i in range(self.N):
        deviations[i,k] = lina.norm(self.X[i,:] - Mu[:,k])/Nks[k]
        wDeviations[i,k] = Gamma[i,k]*deviations[i,k]

    for k in range(self.K):
      sigma[:,:,k] = np.sum(wDeviations[:,k],axis=0)*np.identity(self.D)

    return sigma

  def _sigNewGen(self, eUpdates, Mu):
    Nks = eUpdates.Nks
    Gamma = eUpdates.gamma
    sigma = np.empty([self.D,self.D,self.K])
    sigma_ = np.empty([self.N,self.D,self.D,self.K])
    for i in range(self.N):
      for k in range(self.K):
        dif = self.X[i,:] - Mu[:,k]
        sigma_[i,:,:,k] = Gamma[i,k]*np.outer(dif,dif)/Nks[k]

    for k in range(self.K):
      sigma[:,:,k] = np.sum(sigma_[:,:,:,k],axis=0)

    return sigma

  @staticmethod
  def _groupByLabel(values, labels, K):
    upToN = range(len(values))
    groups = [[values[j] for j in upToN if labels[j] == i] for i in range(K)]
    return groups

  @staticmethod
  def _plotContours(fun, figId, c, alpha_=0.2, pi=1):
    X = np.linspace(-10,13,1000)
    Y = np.linspace(-10,13,1000)
    X,Y = np.meshgrid(X,Y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    Z = pi*fun(pos)
    plt.figure(figId)
    CS = plt.contour(X, Y, Z,levels=10, alpha=alpha_, colors = c)

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

