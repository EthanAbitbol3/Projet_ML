from linear import Linear
from MSELoss import MSELoss
import TanH,Sigmoide,Sequentiel,projet_etu,Optim
from sklearn.datasets import make_blobs,make_moons,make_regression
from matplotlib import pyplot as plt
from pandas import DataFrame
import numpy as np 

# Génération des points
X, y = make_regression(n_samples=100, n_features=1,bias=0.5,noise=10)


def neural_network(X_train, y_train, nombre_neurone, learning_rate=0.0001, n_iter=30, model = Linear, f_cout = MSELoss):
    y_train = y_train.reshape((-1,1))
    modele = model(X_train.shape[1],nombre_neurone)
    cout = f_cout()
    train_loss = []
    train_accuracy = []
    for _ in range(n_iter):
        y_hat = modele.forward(X_train)
        last_delta = cout.backward(y_train,np.sum(y_hat,axis=1))
        #last_delta = cout.backward(y_train,y_hat)
        delta = modele.backward_delta(X_train,last_delta)
        modele.backward_update_gradient(X_train,delta)
        modele.update_parameters(learning_rate)
        modele.zero_grad()
        train_loss.append(cout.forward(y_train,y_hat).mean())
    
    return modele, train_loss

modele, train_loss = neural_network(X,y,1)


# Affichage de l'évolution de l'erreur
plt.figure()
plt.plot(range(len(train_loss)),train_loss)
plt.show()




