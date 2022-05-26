"""
ABITBOL YOSSEF
DUFOURMANTELLE JEREMY

Fichier de test pour Lineaire
"""

import sys
sys.path.insert(1, '../code/')

from Linear import Linear

from loss.MSELoss import MSELoss

from sklearn.datasets import make_regression

from matplotlib import pyplot as plt

# TEST PARTIE 1: LINEAIRE MODULE

def neural_network_lineaire(X_train, y_train, nombre_neurone , n_iter = 200,learning_rate = 0.001):
    # nombre_neurone = y_train.shape[1]
    modele = Linear(X_train.shape[1],nombre_neurone)
    loss = MSELoss()
    train_loss = []

    for _ in range(n_iter):
        # phase forward
        y_hat = modele.forward(X_train)

        # retro propagation du gradient de la loss par rapport aux parametres et aux entrees
        forward_loss = loss.forward(y_train,y_hat)
        print('Erreur moyenne loss :',forward_loss.mean())
        train_loss.append(forward_loss.mean()) # erreur

        backward_loss = loss.backward(y_train,y_hat)

        modele.backward_update_gradient(X_train,backward_loss)

        # mise a jour  de la matrice de poids 
        modele.update_parameters(learning_rate)
        modele.zero_grad() 
    
    return modele, train_loss

def affichage(X,y,modele,train_loss):
    # Affichage de l'évolution de l'erreur
    plt.figure()
    plt.plot(range(len(train_loss)),train_loss,label = 'train_loss')
    plt.legend()
    plt.title('Erreur en fonction de litération')
    plt.xlabel('iterations')
    plt.ylabel('erreur')
    plt.show()

    plt.figure()
    if y.ndim == 1 : 
        y.reshape((-1,1))
        label = "w"+str(0)
        plt.scatter(X,y)
        plt.plot(X,X*modele._parameters[0][0],label=label)
        plt.legend()
    else : 
        for i in range(modele._parameters.shape[1]):
            label = "w"+str(i)
            plt.scatter(X,y[:,i])
            plt.plot(X,X*modele._parameters[0][i],label=label)
            plt.legend()
    plt.xlabel("X")
    plt.ylabel("y")
    plt.show()
    return None

# Génération des points
nombre_dim_y = 1
X, y = make_regression(n_samples=100, n_features=1,bias=0.5,noise=10,n_targets=nombre_dim_y, random_state=0)
if y.ndim == 1 :
    y = y.reshape((-1,1))
modele, train_loss = neural_network_lineaire(X,y,nombre_neurone= y.shape[1])
affichage(X,y,modele,train_loss)
