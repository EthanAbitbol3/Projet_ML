import sys
sys.path.insert(1, '../code/')

"""
Utils
"""
from termcolor import colored
from mltools import plot_data, plot_frontiere, make_grid, gen_arti
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import random
import seaborn as sns
import matplotlib.pyplot as plt

"""
Modules principaux
"""
from Sequentiel import Sequentiel
from Optim import SGD
from Linear import Linear
"""
Activation
"""
from activation.Softmax import Softmax
from activation.Sigmoide import Sigmoide
from activation.TanH import TanH
"""
Loss
"""
from loss.MSELoss import MSELoss
from loss.CELoss import CELoss


class DataGenerator : 
    
    def classif_data(self,n_clusters_per_class=1,n_informative=2,n_samples=100,n_classes=2) : 
        return make_classification(n_classes = n_classes,n_features=2, n_samples=n_samples,n_redundant=0, n_informative=n_informative, n_clusters_per_class=n_clusters_per_class)

def print_ok():
    print(colored('OK','green'))

def print_ko():
    print(colored('KO','red'))


def onehot_to_vector(y):
    res = []
    for i in range(y.shape[0]):
        res.append(np.argmax(y[i]))
    res = np.array(res)
    return res.reshape((-1,1))

def onehot(y):
    onehot = np.zeros((y.size, y.max() + 1))
    onehot[np.arange(y.size), y] = 1
    return onehot


## TEST SUR USPS

def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

uspsdatatrain = "../data/USPS_train.txt"
uspsdatatest = "../data/USPS_test.txt"
alltrainx, alltrainy = load_usps(uspsdatatrain)
alltestx, alltesty = load_usps(uspsdatatest)


print(alltrainx.shape)

y = onehot(alltrainy)

linear1 = Linear(alltrainx.shape[1], 256)
activation1 = TanH()
linear2 = Linear(256, 128)
activation2 = TanH()
linear3 = Linear(128, y.shape[1])
activation3 = Softmax()
loss = CELoss()

# Optimization

# Hyperparameters
maxIter = 500
eps = 1e-2
batch_size = 10


model = Sequentiel(Linear(alltrainx.shape[1], 4) , TanH(), Linear(4, y.shape[1]), Softmax())

optimizer = SGD(model, loss, alltrainx, y, batch_size=batch_size, eps=eps, nbIter=maxIter)
list_loss = optimizer.update()

taux_train = ((np.argmax( optimizer.net.forward(alltrainx),axis = 1) == alltrainy).mean()*100)
taux_test = ((np.argmax( optimizer.net.forward(alltestx),axis = 1) == alltesty).mean()*100)
print("Taux de bonne classification en train : ",taux_train,"%")
print("Taux de bonne classification en test : ",taux_test,"%")

"""
AFFICHAGE DE LA LOSS
"""
plt.figure()
plt.xlabel("nombre d'iteration")
plt.ylabel("Erreur CE")
plt.title("Evolution de l'erreur")
plt.plot(list_loss,label="Erreur")
plt.legend()
plt.show()


predict = model.forward(alltrainx)
predict = np.argmax(predict, axis=1)


"""
MATRICE DE CONFUSION TRAIN
"""

plt.figure()
confusion = confusion_matrix(predict, alltrainy)



ax = sns.heatmap(confusion, annot=True, cmap='Blues')

ax.set_title(f"Matrice de confusion pour données USPS Train \ acc = {taux_train}%\n\n")
ax.set_xlabel('\nChiffre prédit')
ax.set_ylabel('Vrai chiffre ')

ax.xaxis.set_ticklabels(np.arange(10))
ax.yaxis.set_ticklabels(np.arange(10))

plt.show()

"""
MATRICE DE CONFUSION TEST
"""


predict = model.forward(alltestx)
predict = np.argmax(predict, axis=1)

plt.figure()
confusion = confusion_matrix(predict, alltesty)

ax = sns.heatmap(confusion, annot=True, cmap='Blues')

ax.set_title(f"Matrice de confusion pour données USPS Test \ acc = {taux_test}%\n\n")
ax.set_xlabel('\nChiffre prédit')
ax.set_ylabel('Vrai chiffre ')

ax.xaxis.set_ticklabels(np.arange(10))
ax.yaxis.set_ticklabels(np.arange(10))

plt.show()

