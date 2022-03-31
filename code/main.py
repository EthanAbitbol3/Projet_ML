from turtle import backward

from sklearn.model_selection import learning_curve
from linear import Linear
from MSELoss import MSELoss
import TanH,Sigmoide,Sequentiel,projet_etu,Optim
from sklearn.datasets import make_blobs,make_moons,make_regression
from matplotlib import pyplot as plt
from pandas import DataFrame
import numpy as np 


# TEST PARTIE 1: LINEAIRE MODULE

def neural_network(X_train, y_train, nombre_neurone, n_iter = 100,learning_rate = 0.001):
    modele = Linear(X_train.shape[1],nombre_neurone)
    loss = MSELoss()
    train_loss = []

    for _ in range(n_iter):
        forward = modele.forward(X_train)
        forward_loss = loss.forward(y_train,forward)
        train_loss.append(forward_loss.sum())
        backward_loss = loss.backward(y_train,forward)   
        modele.backward_update_gradient(X_train,backward_loss)
        modele.update_parameters(learning_rate)
        modele.zero_grad() 
    
    return modele, train_loss

def affichage(modele,train_loss):
    # Affichage de l'évolution de l'erreur
    plt.figure()
    plt.plot(range(len(train_loss)),train_loss,label = 'train_loss')
    plt.legend()
    plt.title('Erreur en fonction de litération')
    plt.xlabel('iterations')
    plt.ylabel('erreur')
    plt.show()

    plt.figure()
    for i in range(modele._parameters.shape[1]):
        label = "w"+str(i)
        plt.scatter(X,y[:,i])
        plt.plot(X,X*modele._parameters[0][i],label=label)
        plt.legend()
    plt.xlabel("X")
    plt.ylabel("y")
    plt.show()

# Génération des points
X, y = make_regression(n_samples=100, n_features=1,bias=0.5,noise=10,n_targets=5, random_state=0)
modele, train_loss = neural_network(X,y,5)
affichage(modele,train_loss)




