"""
SEQUENTIEL
"""


import sys
sys.path.insert(1, '../code/')


from loss.MSELoss import MSELoss
from activation.Sigmoide import Sigmoide
from activation.TanH import TanH
from Linear import Linear
import numpy as np
from mltools import plot_data, plot_frontiere, gen_arti
import matplotlib.pyplot as plt
from Sequentiel import Sequentiel

from tqdm import tqdm 


# generations de points 
np.random.seed(1)

X, y = gen_arti(data_type=1, epsilon=0.001)

if y.ndim == 1 : 
    y = y.reshape((-1,1))
y = np.where(y > 0 , 1  , 0)

nombre_neurone = 2
nn = Sequentiel(Linear(X.shape[1],nombre_neurone),TanH(),Linear(nombre_neurone,y.shape[1]),Sigmoide())
loss = MSELoss()
nbIter = 500
list_loss = []
learning_rate = 0.01

for i in tqdm(range(nbIter)):
    y_hat = nn.forward(X)
    last_delta = loss.backward(y, y_hat)
    delta = nn.backward_delta(X, last_delta)
    nn.backward_update_gradient(X,delta)
    nn.update_parameters(learning_rate)
    nn.zero_grad()
    _loss = np.mean(loss.forward(y, y_hat))
    list_loss.append(_loss)
    print("Loss : ",_loss)


acc = np.where(y == nn.predict(X),1,0).mean()

# affichage de la frontiere de decision ainsi que des donnees
plt.figure()
plot_frontiere(X,nn.predict )
plot_data(X,y)
plt.title(f"Sequentiel avec {nombre_neurone} neurones / acc = {(acc*100)}%")
plt.show()

print((np.where(nn.predict(X)>=0.5,1,0) == y ).mean())
    
print("MSE : ",np.mean(loss.forward(y, y_hat)))
print("Taux de bonne classification : ",((nn.predict(X) == np.where(y>=0.5,1,0)).sum()/len(np.where(y>=0.5,1,0)))*100,"%")
# affichage erreur 
plt.figure()
plt.xlabel("nombre d'iteration")
plt.ylabel("Erreur MSE")
plt.title("Evolution de l'erreur")
plt.plot(np.arange(nbIter),list_loss,label="Erreur")
plt.legend()
plt.show()