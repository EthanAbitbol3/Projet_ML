import sys
sys.path.insert(1, '../code/')

from loss.BCELoss import BCELoss
from loss.CELoss import CELoss
from loss.MSELoss import MSELoss
from activation.TanH import TanH
from activation.Sigmoide import Sigmoide
from activation.Softmax import Softmax
from Linear import Linear
from Optim import SGD
from Sequentiel import Sequentiel
from sklearn.preprocessing import StandardScaler
import random
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.datasets import make_classification
from mltools import plot_data, plot_frontiere, make_grid, gen_arti
from termcolor import colored
import tensorflow as tf
import sys
import pandas as pd
sys.path.insert(1, '../code/')


"""
Utils
"""

"""
Modules principaux
"""
"""
Activation
"""
"""
Loss
"""


# TEST SUR USPS

def load_usps(fn):
    with open(fn, "r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split()) > 2]
    tmp = np.array(data)
    return tmp[:, 1:], tmp[:, 0].astype(int)

def onehot(y):
    onehot = np.zeros((y.size, y.max() + 1))
    onehot[np.arange(y.size), y] = 1
    return onehot



def test_autoEncoder(transform1 = 512, tranform2 = 10):
    nom_fichier_train =  "../data/mnist_train.csv"
    nom_fichier_test =  "../data/mnist_test.csv"
    data_train =  pd.read_csv(nom_fichier_train).to_numpy()
    data_test =  pd.read_csv(nom_fichier_test).to_numpy()
    alltrainx,alltrainy = data_train[:,1:].astype('float32') , data_train[:,0]
    alltestx, alltesty = data_test[:,1:].astype('float32') , data_test[:,0]
    alltrainx /= 255
    alltestx /= 255

    _y = alltrainy
    alltrainy = onehot(alltrainy)

    iteration = 100
    eps = 1e-3
    batch_size = 20
    l1 = Linear(alltrainx.shape[1], transform1)
    l2 = Linear(transform1, tranform2)
    l3 = Linear(tranform2, transform1)
    l4 = Linear(transform1, alltrainx.shape[1])
    l3._parameters = l2._parameters.T.copy()
    l4._parameters = l1._parameters.T.copy()

    encoder = Sequentiel(l1, TanH(), l2, TanH())
    decoder = Sequentiel(l3, TanH(), l4, Sigmoide())
    model = Sequentiel(encoder, decoder)
    loss = BCELoss()
    opt = SGD(model, loss, alltrainx, alltrainx,
              batch_size, nbIter=iteration, eps=eps)
    opt.update()


    X_hat = model.forward(alltrainx)

    # Affichage des données originelles
    plt.figure(figsize=(20, 4))
    n = 10
    plt.title(f"originales : c1 = {transform1} / c2 = {tranform2}")
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)

        plt.imshow(alltrainx[i].reshape(28, 28))
        plt.gray()

    plt.show()

    # Affichage des images codé/décodé
    n = 10
    plt.figure(figsize=(20, 4))
    plt.title(f"E/D : c1 = {transform1} / c2 = {tranform2}")
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_hat[i].reshape(28, 28))
        plt.gray()
    plt.show()




if __name__ == '__main__':
    np.random.seed(0)
    # Test 1 : test de base avec les valeurs du sujet
    test_autoEncoder()

    

    # # Test 2 : 50 -> 5
    # test_autoEncoder(transform1 = 50, tranform2 = 5)

    # # Test 3 : 200 -> 20
    # test_autoEncoder(transform1 = 200, tranform2 = 20)

    # # Test 4 : 20 -> 2
    # test_autoEncoder(transform1 = 20, tranform2 = 2)