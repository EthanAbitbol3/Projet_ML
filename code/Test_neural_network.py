from Optim import Optim
import unittest
from termcolor import colored
from mltools import plot_data, plot_frontiere, make_grid, gen_arti
from sklearn.datasets import make_classification
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from Sequentiel import Sequentiel
from MSELoss import MSELoss
from Sigmoide import Sigmoide
from TanH import Tanh
from linear import Linear
from Optim import Optim
from Optim import SGD
import time
from tqdm import tqdm

def print_ok():
    print(colored('OK','green'))

def print_ko():
    print(colored('KO','red'))

class DataGenerator : 
    
    def classif_data(self,n_clusters_per_class=1,n_informative=1,n_samples=100) : 
        return make_classification(n_features=2, n_samples=n_samples,n_redundant=0, n_informative=n_informative, n_clusters_per_class=n_clusters_per_class)

class TestNeuralNetwork : 

    def __init__(self,X,y,network,loss):
        self.X = X
        self.y = y
        self.network = network
        self.loss = loss
        self.list_error = []
        self.time_learning = 0

    """
    Optimizer Batch Gradient Descent
    """
    def batchGradientDescent(self,eps = 1e-3, max_iter = 1000) :
        print("-- Descente de gradient en batch sur",str(self.X.shape[0]),"exemples --")
        optim = Optim(self.network , self.loss , eps)
        time_start = time.time()
        for _ in tqdm(range(max_iter)) :
            self.list_error.append(optim.step(self.X,self.y))
        time_end = time.time()
        self.time_learning = time_end - time_start
    """
    Optimizer Stochastic Gradient Descent
    """
    def stochasticGradientDescent(self,eps = 1e-3, max_iter = 1000):
        print("-- Descente de gradient stochastique sur",str(self.X.shape[0]),"exemples --")
        optim = Optim(self.network , self.loss , eps)
        X,y = self.X,self.y
        time_start = time.time()
        for _ in tqdm(range(max_iter)) :
            X , y = shuffle(X,y, random_state=1)
            for i in range(X.shape[0]):
                _y = y[i]
                _x = np.array([[X[i][j] for j in range(len(X[i]))]])
                self.list_error.append(optim.step(_x,_y))
        time_end = time.time()
        self.time_learning = time_end - time_start

    """
    Optimizer Mini Batch Gradient Descent
    """

    def MiniBatchGradientDescent(self,eps = 1e-3, max_iter = 1000):
        pass


    """
    Affichage
    """
    def printResult(self): 
        y_hat = self.network.forward(self.X)
        print("Moyenne des erreurs : ",np.mean(loss.forward(self.y, y_hat)))
        print("Taux de bonne classification : ",((self.network.predict(self.X,biais = False) == np.where(self.y>=0.5,1,0)).sum()/len(self.y))*100,"%")
        print("Durée d'apprentissage : ",self.time_learning,"ms")
        print_ok()

    def plotResult(self) :
        """
        plt.figure()
        plot_frontiere(X,self.network.predict)
        plot_data(X,y)
        plt.title(f"neural network non lineaire avec {nombre_neurone} neurones")
        plt.show()
        """
        

#################################################################################
###############################   TEST   ########################################
#################################################################################

# TEST 1 
dg = DataGenerator() # generateur de données de classification
X,y = dg.classif_data(n_samples=1000) # génération des données
if y.ndim == 1 : 
    y = y.reshape((-1,1))

# Affichage des données
plt.figure()
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()


# MODELE 1 
eps = 1e-5
nombre_neurone = 4
modules = [Linear(X.shape[1],nombre_neurone),Tanh(),Linear(nombre_neurone,y.shape[1]),Sigmoide()]
loss = MSELoss()
network = Sequentiel(modules)
tnn = TestNeuralNetwork(X,y,network,loss) # Création du test
tnn.stochasticGradientDescent() # descente de gradient stocastique
tnn.printResult() # affichage dans le terminal des résultats
tnn.plotResult() # affichage de la frontiere de décision

# MODELE 2
nombre_neurone = 4
modules = [Linear(X.shape[1],nombre_neurone),Tanh(),Linear(nombre_neurone,y.shape[1]),Sigmoide()]
loss = MSELoss()
network = Sequentiel(modules)
tnn = TestNeuralNetwork(X,y,network,loss) # Création du test
tnn.batchGradientDescent() # descente de gradient en batch
tnn.printResult() # affichage dans le terminal des résultats
tnn.plotResult() # affichage de la frontiere de décision
