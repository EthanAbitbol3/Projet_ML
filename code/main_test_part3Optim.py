import numpy as np
from Sequentiel import Sequentiel
from MSELoss import MSELoss
from Sigmoide import Sigmoide
from TanH import Tanh
from linear import Linear
from Optim import Optim
from Optim import SGD

import numpy as np
from mltools import plot_data, plot_frontiere, make_grid, gen_arti
import matplotlib.pyplot as plt

"""
TEST OPTIM 
"""

# GENERATION DES DONNEES
np.random.seed(1)
X, y = gen_arti(data_type=1, epsilon=0.001) # 4 gaussiennes
bias = np.ones((len(X), 1))
Xbiais = np.hstack((bias, X))
if y.ndim == 1 : 
    y = y.reshape((-1,1))

# PARAMETRES DU MODELE
eps = 1e-10
nombre_neurone = 4
modules = [Linear(Xbiais.shape[1],nombre_neurone),Tanh(),Linear(nombre_neurone,y.shape[1]),Sigmoide()]
#modules = [Linear(Xbiais.shape[1],nombre_neurone),Tanh(),Linear(nombre_neurone,nombre_neurone),Tanh(),
#Linear(nombre_neurone,nombre_neurone),Tanh(),Linear(nombre_neurone,y.shape[1]),Sigmoide()]
loss = MSELoss()
network = Sequentiel(modules)
optim = Optim(network , loss , eps)
nbIter = 1000


# TEST POUR LA CLASSE OPTIM (a decommenter)
"""
list_loss = []
for i in range(nbIter):
    list_loss.append(optim.step(Xbiais,y))
"""

# TEST POUR LA FONCTION SGD
batch_size = 500
neural_network_trained,list_loss = SGD(network , loss , Xbiais , y , batch_size , nbIter , eps)

print(Xbiais)
# affichage de la frontiere de decision ainsi que des donnees
plt.figure()
plot_frontiere(X,neural_network_trained.predict)
plot_data(X,y)
plt.title(f"Frontiere de decision du module SGD optimisation avec {nombre_neurone} neurones")
plt.show()

# affichage erreur 
plt.figure()
plt.xlabel("nombre d'iteration")
plt.ylabel("Erreur MSE")
plt.title("Evolution de l'erreur SGD optimisation")
plt.plot(np.arange(nbIter),list_loss,label="Erreur")
plt.legend()
plt.show()