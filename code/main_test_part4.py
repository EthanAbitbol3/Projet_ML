from Optim import Optim
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
from Optim import SGD
import time
import random
from tqdm import tqdm
from LogSoftmax import *
from Softmax import *
from CELoss import *
from LogCELoss import *

class DataGenerator : 
    
    def classif_data(self,n_clusters_per_class=1,n_informative=2,n_samples=100,n_classes=2) : 
        return make_classification(n_classes = n_classes,n_features=2, n_samples=n_samples,n_redundant=0, n_informative=n_informative, n_clusters_per_class=n_clusters_per_class)

def print_ok():
    print(colored('OK','green'))

def print_ko():
    print(colored('KO','red'))

class TestNeuralNetwork : 

    def __init__(self,X,y,network,loss):
        self.X = X
        self.y = y
        self.network = network
        self.loss = loss
        self.list_error = []
        self.time_learning = 0

    """
    Optimizer Batch Gradient Descent (BGD)
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
    Optimizer Stochastic Gradient Descent (SGD)
    """
    def stochasticGradientDescent(self,eps = 1e-3, max_iter = 1000):
        print("-- Descente de gradient stochastique sur",str(self.X.shape[0]),"exemples --")
        optim = Optim(self.network , self.loss , eps)
        print("test 1 : ",self.X.shape)
        X,y = self.X,self.y
        time_start = time.time()
        for _ in tqdm(range(max_iter)) :
            #X , y = shuffle(X,y, random_state=1)
            i = random.randint(0,(len(y)-1))
            _y = y[i]
            _x = np.array([[X[i][j] for j in range(len(X[i]))]])
            self.list_error.append(optim.step(_x,_y))
        time_end = time.time()
        self.time_learning = time_end - time_start

    """
    Optimizer Mini Batch Gradient Descent (MBGD)
    """
    def MiniBatchGradientDescent(self,eps = 1e-3, max_iter = 1000):
        pass


    """
    Affichage
    """
    def printResult(self):
        print("test 2 : ",self.X.shape) 
        y_hat = self.network.forward(self.X)
        print(y_hat.shape)
        print("Moyenne des erreurs : ",np.mean(self.loss.forward(self.y, y_hat)))
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

# TEST 1 
dg = DataGenerator() # generateur de données de classification
X,y = dg.classif_data(n_samples=1000,n_classes=3) # génération des données
if y.ndim == 1 : 
    y = y.reshape((-1,1))

def onehot(y):
    new_y = []
    classes = np.unique(y)
    for _y in y:
        classe = np.zeros(len(classes))
        classe[_y] = 1
        new_y.append(classe)
    return np.array(new_y)

y = onehot(y)

# Affichage des données
plt.figure()
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()

# MODELE 1 
eps = 1e-5
nombre_neurone = 4
modules = [Linear(X.shape[1],nombre_neurone),Tanh(),Linear(nombre_neurone,y.shape[1]),LogSoftmax()]
loss = LogCELoss()
print(modules)

network = Sequentiel(modules)
tnn = TestNeuralNetwork(X,y,network,loss) # Création du test
tnn.batchGradientDescent() # descente de gradient en batch
tnn.printResult() # affichage dans le terminal des résultats

"""
loss = MSELoss()
network = Sequentiel(modules)
tnn = TestNeuralNetwork(X,y,network,loss) # Création du test
tnn.stochasticGradientDescent() # descente de gradient stocastique
tnn.printResult() # affichage dans le terminal des résultats
tnn.plotResult() # affichage de la frontiere de décision
"""