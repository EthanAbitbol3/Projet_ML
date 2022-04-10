from MSELoss import MSELoss
from Sigmoide import Sigmoide
from TanH import Tanh
from linear import Linear
import numpy as np
from projet_etu import Module 
from mltools import plot_data, plot_frontiere, make_grid, gen_arti
import matplotlib.pyplot as plt
from Sequentiel import Sequentiel

# generations de points 
np.random.seed(1)
X, y = gen_arti(data_type=1, epsilon=0.001) # 4 gaussiennes
bias = np.ones((len(X), 1))
Xbiais = np.hstack((bias, X))

if y.ndim == 1 : 
    y = y.reshape((-1,1))

nombre_neurone = 4

modules = [Linear(Xbiais.shape[1],nombre_neurone),Tanh(),Linear(nombre_neurone,y.shape[1]),Sigmoide()]
#modules = [Linear(Xbiais.shape[1],nombre_neurone),Tanh(),Linear(nombre_neurone,nombre_neurone),Tanh(),
#Linear(nombre_neurone,nombre_neurone),Tanh(),Linear(nombre_neurone,y.shape[1]),Sigmoide()]
loss = MSELoss()
nn = Sequentiel(modules)

nbIter = 200
list_loss = []

for i in range(nbIter):
    y_hat = nn.forward(Xbiais)
    last_delta = loss.backward(y, y_hat)
    delta = nn.backward_delta(Xbiais, last_delta)
    nn.backward_update_gradient(Xbiais,delta)
    nn.update_parameters()
    nn.zero_grad()
    nn.initialisation_parameters()
    list_loss.append(np.sum(loss.forward(y, y_hat)))
    print('Loss :',np.sum(loss.forward(y, y_hat)))

# affichage de la frontiere de decision ainsi que des donnees
plt.figure()
plot_frontiere(X,nn.predict)
plot_data(X,y)
plt.title(f"Frontiere de decision du module Sequence avec {nombre_neurone} neurones")
plt.show()

# affichage erreur 
plt.figure()
plt.xlabel("nombre d'iteration")
plt.ylabel("Erreur MSE")
plt.title("Evolution de l'erreur")
plt.plot(np.arange(nbIter),list_loss,label="Erreur")
plt.legend()
plt.show()


class neural_network_non_lineaire:
    def __init__(self):
        self.list_error = []

    def fit(self,X, y, nombre_neurone, n_iter = 100 , learning_rate = 0.001, biais = True):
        if biais == True:
            bias = np.ones((len(X), 1))
            X = np.hstack((bias, X))

        # Initialisation des modules
        self.mse = MSELoss()
        self.linear_1 = Linear(X.shape[1], nombre_neurone)
        self.tanh = Tanh()
        self.linear_2 = Linear(nombre_neurone, y.shape[1])
        self.sigmoide = Sigmoide()

        for _ in range(n_iter):
            
            # phase forward
            res1 = self.linear_1.forward(X)
            res2 = self.tanh.forward(res1)
            res3 = self.linear_2.forward(res2)
            res4 = self.sigmoide.forward(res3)

            self.list_error.append(np.sum(self.mse.forward(y, res4))) # loss
            print('Loss :',np.sum(self.mse.forward(y, res4)))
            #  retro propagation du gradient de la loss par rapport aux parametres et aux entrees
            last_delta = self.mse.backward(y, res4)

            delta_sig = self.sigmoide.backward_delta(res3, last_delta)
            delta_lin = self.linear_2.backward_delta(res2, delta_sig)
            delta_tan = self.tanh.backward_delta(res1, delta_lin)

            self.linear_1.backward_update_gradient(X, delta_tan)
            self.linear_2.backward_update_gradient(res2, delta_sig)

            # mise a jour  de la matrice de poids
            self.linear_1.update_parameters(learning_rate)
            self.linear_2.update_parameters(learning_rate)

            self.linear_1.zero_grad()
            self.linear_2.zero_grad()

    def predict(self,xtest,biais = True):
        if biais == True:
            bias = np.ones((len(xtest), 1))
            xtest = np.hstack((bias, xtest))
        
        res1 = self.linear_1.forward(xtest)
        res2 = self.tanh.forward(res1)
        res3 = self.linear_2.forward(res2)
        res4 = self.sigmoide.forward(res3)

        return np.where(res4>=0.5,1,0) 


# generations de points 

neural_network_non_lineaire = neural_network_non_lineaire()
neural_network_non_lineaire.fit(X,y,nombre_neurone=nombre_neurone,n_iter=200,learning_rate=0.001)

# affichage de la frontiere de decision ainsi que des donnees
plt.figure()
plot_frontiere(X,neural_network_non_lineaire.predict)
plot_data(X,y)
plt.title(f"neural network non lineaire avec {nombre_neurone} neurones")
plt.show()

# affichage de la courbe d'errreur
plt.figure()
plt.title('erreur en fonction de literation')
plt.plot(neural_network_non_lineaire.list_error, label='loss')
plt.legend()
plt.xlabel('itérations')
plt.ylabel('erreur')
plt.show()