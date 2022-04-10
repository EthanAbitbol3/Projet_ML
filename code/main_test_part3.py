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
<<<<<<< HEAD
if y.ndim == 1 : 
    y = y.reshape((-1,1))
=======

if y.ndim == 1 : 
    y = y.reshape((-1,1))

>>>>>>> af604571a72bded991c3dc924a2ffceec5de8eb9
nombre_neurone = 4

modules = [Linear(Xbiais.shape[1],nombre_neurone),Tanh(),Linear(nombre_neurone,y.shape[1]),Sigmoide()]
#modules = [Linear(Xbiais.shape[1],nombre_neurone),Tanh(),Linear(nombre_neurone,nombre_neurone),Tanh(),
#Linear(nombre_neurone,nombre_neurone),Tanh(),Linear(nombre_neurone,y.shape[1]),Sigmoide()]
loss = MSELoss()
nn = Sequentiel(modules)

<<<<<<< HEAD
nbIter = 1000
list_loss = []
=======
nbIter = 200
list_loss = []

>>>>>>> af604571a72bded991c3dc924a2ffceec5de8eb9
for i in range(nbIter):
    y_hat = nn.forward(Xbiais)
    last_delta = loss.backward(y, y_hat)
    delta = nn.backward_delta(Xbiais, last_delta)
    nn.backward_update_gradient(Xbiais,delta)
    nn.update_parameters()
    nn.zero_grad()
    nn.initialisation_parameters()
    list_loss.append(np.sum(loss.forward(y, y_hat)))
<<<<<<< HEAD

=======
    print('Loss :',np.sum(loss.forward(y, y_hat)))

# affichage de la frontiere de decision ainsi que des donnees
plt.figure()
plot_frontiere(X,nn.predict)
plot_data(X,y)
plt.title(f"Frontiere de decision du module Sequence avec {nombre_neurone} neurones")
plt.show()

# affichage erreur 
>>>>>>> af604571a72bded991c3dc924a2ffceec5de8eb9
plt.figure()
plt.xlabel("nombre d'iteration")
plt.ylabel("Erreur MSE")
plt.title("Evolution de l'erreur")
plt.plot(np.arange(nbIter),list_loss,label="Erreur")
plt.legend()
<<<<<<< HEAD
plt.show()

# affichage de la frontiere de decision ainsi que des donnees
print(X.shape)
plt.figure()
plot_frontiere(X,nn.predict)
plot_data(X,y)
plt.title(f"Frontiere de décision du module Sequence avec {nombre_neurone} neurones")
=======
>>>>>>> af604571a72bded991c3dc924a2ffceec5de8eb9
plt.show()