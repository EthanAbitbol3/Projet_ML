##########################################################################################################
##########################################################################################################
##########################################################################################################
################################# ABITBOL YOSSEF ET DUFOURMANTELLE JEREMY ################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

# iMPORT 
from sys import modules
from linear import Linear
from MSELoss import MSELoss
from TanH import Tanh
from Sigmoide import Sigmoide
from Sequentiel import Sequentiel
from Optim import Optim, SGD
from Softmax import Softmax
from LogSoftmax import LogSoftmax
from CELoss import CELoss
from LogCELoss import LogCELoss

from sklearn.datasets import make_blobs,make_moons,make_regression
from matplotlib import pyplot as plt
import numpy as np 
from mltools import plot_data, plot_frontiere, make_grid, gen_arti
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# UTILITAIRE #

def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def get_usps(l,datax,datay):
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy
    tmp =   list(zip(*[get_usps(i,datax,datay) for i in l]))
    tmpx,tmpy = np.vstack(tmp[0]),np.hstack(tmp[1])
    return tmpx,tmpy

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")

def OneHotEncoding(y):
    onehot = np.zeros((y.size, y.max() + 1))
    onehot[np.arange(y.size), y] = 1
    return onehot
    
##########################################################################################################
##########################################################################################################
##########################################################################################################

# TEST PARTIE 1: LINEAIRE MODULE

def neural_network_lineaire(X_train, y_train, nombre_neurone , n_iter = 100,learning_rate = 0.001):
    # nombre_neurone = y_train.shape[1]
    modele = Linear(X_train.shape[1],nombre_neurone)
    loss = MSELoss()
    train_loss = []

    for _ in range(n_iter):
        # phase forward
        y_hat = modele.forward(X_train)

        # retro propagation du gradient de la loss par rapport aux parametres et aux entrees
        forward_loss = loss.forward(y_train,y_hat)
        print('Loss :',forward_loss.sum())
        train_loss.append(forward_loss.sum()) # erreur

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

"""
# Génération des points
nombre_dim_y = 2
X, y = make_regression(n_samples=100, n_features=1,bias=0.5,noise=10,n_targets=nombre_dim_y, random_state=0)
if y.ndim == 1 : 
    y = y.reshape((-1,1))
modele, train_loss = neural_network_lineaire(X,y,nombre_neurone= y.shape[1])
affichage(X,y,modele,train_loss)
"""

##########################################################################################################
##########################################################################################################
##########################################################################################################

# TEST PARTIE 2: NON LINEAIRE MODULE

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

"""
# generations de points 
np.random.seed(1)
X, y = gen_arti(data_type=1, epsilon=0.001) # 4 gaussiennes
if y.ndim == 1 : 
    y = y.reshape((-1,1))
nombre_neurone = 4
neural_network_non_lineaire = neural_network_non_lineaire()
neural_network_non_lineaire.fit(X,y,nombre_neurone=nombre_neurone,n_iter=200,learning_rate=0.01)

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
"""

##########################################################################################################
##########################################################################################################
##########################################################################################################

# TEST PARTIE 3: Mon troisième est un encapsulage : SEQUENTIEL
"""
# generations de points 
X2, y2 = gen_arti(data_type=1, epsilon=0.01) # 4 gaussiennes

# preprocessing
scaler = StandardScaler()
X2 = scaler.fit_transform(X2)

# classification sur 0 et 1
y2 = np.array([ 0 if d == -1 else 1 for d in y2 ])
onehot = np.zeros((y2.size, 2))
onehot[np.arange(y2.size), y2 ] = 1
y2 = onehot

###### FIT 
nombre_neurone = 12

mse = MSELoss()
linear_1 = Linear(X2.shape[1], nombre_neurone)
tanh = Tanh()
linear_2 = Linear(nombre_neurone, y2.shape[1])
sigmoide = Sigmoide()

modules = [linear_1,tanh,linear_2,sigmoide]
for _ in range(1000):
    sequentiel = Sequentiel(modules,mse)
    sequentiel.fit(X2, y2)

    linear_1.backward_update_gradient(X2, sequentiel.delta[-1])
    linear_2.backward_update_gradient(sequentiel.res[1], sequentiel.delta[1])

    # mise a jour  de la matrice de poids
    linear_1.update_parameters(0.001)
    linear_2.update_parameters(0.001)

    linear_1.zero_grad()
    linear_2.zero_grad()

y2 = np.argmax(y2,axis=1)

# affichage de la frontiere de decision ainsi que des donnees
plt.figure()
plot_frontiere(X2,lambda x : sequentiel.predict(x,biais = False),step=100)
plot_data(X2,y2.reshape(1,-1)[0])
plt.title(f"Sequentiel avec {nombre_neurone} neurones")
plt.show()

# affichage de la courbe d'errreur
plt.figure()
plt.title('erreur en fonction de literation')
plt.plot(np.arange(len(sequentiel.list_error)),sequentiel.list_error, label='loss')
plt.legend()
plt.xlabel('itérations')
plt.ylabel('erreur')
plt.show()
"""
# TEST PARTIE 3 : OPTIM / SGD

"""
# generations de points 
X, y = gen_arti(data_type=1, epsilon=0.01) # 4 gaussiennes

# preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)

# classification sur 0 et 1
y = np.array([ 0 if d == -1 else 1 for d in y ])
onehot = np.zeros((y.size, 2))
onehot[np.arange(y.size), y ] = 1
y = onehot

nombre_neurone = 5
mse = MSELoss()
linear_1 = Linear(X.shape[1], nombre_neurone)
tanh = Tanh()
linear_2 = Linear(nombre_neurone, y.shape[1])
sigmoide = Sigmoide()
net = [linear_1,tanh,linear_2,sigmoide]

sgd = SGD(net,mse,X,y)
sgd.fit(taille_batch=len(y))

y = np.argmax(y,axis=1)

# affichage de la frontiere de decision ainsi que des donnees
plt.figure()
plot_frontiere(X,lambda x : sgd.predict(x),step=100)
plot_data(X,y.reshape(1,-1)[0])
plt.title(f"neural network non lineaire avec {nombre_neurone} neurones")
plt.show()

# affichage de la courbe d'errreur
plt.figure()
plt.title('erreur en fonction de literation')
plt.plot(sgd.list_error, label='loss')
plt.legend()
plt.xlabel('itérations')
plt.ylabel('erreur')
plt.show()
"""
##########################################################################################################
##########################################################################################################
##########################################################################################################

# TEST PARTIE 4
# softmax et log softmax

class soft():
    def __init__(self):
        self.list_error = []

    def fit(self,X, y, nombre_neurone, n_iter = 100 , learning_rate = 0.001, biais = True):
        if biais == True:
            bias = np.ones((len(X), 1))
            X = np.hstack((bias, X))

        # Initialisation des modules
        self.ce = LogCELoss()
        self.linear_1 = Linear(X.shape[1], nombre_neurone)
        self.tanh = Tanh()
        self.linear_2 = Linear(nombre_neurone, y.shape[1])
        self.softmax = Softmax()

        for _ in range(n_iter):

            # phase forward
            res1 = self.linear_1.forward(X)
            res2 = self.tanh.forward(res1)
            res3 = self.linear_2.forward(res2)
            res4 = self.softmax.forward(res3)

            self.list_error.append(np.mean(self.ce.forward(y, res4))) # loss
            print('Loss :',np.mean(self.ce.forward(y, res4)))
            #  retro propagation du gradient de la loss par rapport aux parametres et aux entrees
            last_delta = self.ce.backward(y, res4)

            delta_sig = self.softmax.backward_delta(res3, last_delta)
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
        res4 = self.softmax.forward(res3)

        # return np.where(res4>=0.5,1,0)
        return np.argmax(res4, axis=1)

"""
uspsdatatrain = "../data/USPS_train.txt"
uspsdatatest = "../data/USPS_test.txt"
alltrainx,alltrainy = load_usps(uspsdatatrain)
alltestx,alltesty = load_usps(uspsdatatest)

alltrainy = OneHotEncoding(alltrainy)

# generations de points 
np.random.seed(1)

if alltrainy.ndim == 1 : 
    alltrainy = alltrainy.reshape((-1,1))

nombre_neurone = 10
soft = soft()
soft.fit(alltrainx,alltrainy,nombre_neurone=nombre_neurone,n_iter=100,learning_rate=0.0001)

# affichage de la courbe d'errreur
plt.figure()
plt.title('erreur en fonction de literation')
plt.plot(soft.list_error, label='loss')
plt.legend()
plt.xlabel('itérations')
plt.ylabel('erreur')
plt.show()

# Test sur les données d'apprentissage
# ypred = soft.predict(alltestx)
# print("Score de bonne classification: ", 1 - np.mean( ypred == alltesty ))
"""
#################### TEST 2 ############################
class soft2():
    def __init__(self):
        self.list_error = []

    def fit(self,X, y, nombre_neurone, n_iter = 100 , learning_rate = 0.001, biais = True):
        if biais == True:
            bias = np.ones((len(X), 1))
            X = np.hstack((bias, X))

        # Initialisation des modules
        self.ce = LogCELoss()
        self.linear_1 = Linear(X.shape[1], nombre_neurone)
        self.tanh = Tanh()
        self.linear_2 = Linear(nombre_neurone, 5)
        self.tanh2 = Tanh()
        self.linear_3 = Linear(5, y.shape[1])
        self.softmax = Softmax()

        for _ in range(n_iter):

            # phase forward
            res1 = self.linear_1.forward(X)
            res2 = self.tanh.forward(res1)
            res3 = self.linear_2.forward(res2)
            res4 = self.tanh2.forward(res3)
            res5 = self.linear_3.forward(res4)
            res6 = self.softmax.forward(res5)

            self.list_error.append(np.mean(self.ce.forward(y, res6))) # loss
            print('Loss :',np.mean(self.ce.forward(y, res6)))
            #  retro propagation du gradient de la loss par rapport aux parametres et aux entrees
            last_delta = self.ce.backward(y, res6)

            delta_soft = self.softmax.backward_delta(res5, last_delta)
            delta_lin3 = self.linear_3.backward_delta(res4, delta_soft)
            delta_tan2 = self.tanh.backward_delta(res3, delta_lin3)
            delta_lin2 = self.linear_2.backward_delta(res2, delta_tan2)
            delta_tan = self.tanh.backward_delta(res1, delta_lin2)
            delta_lin1 = self.linear_1.backward_delta(X, delta_tan)

            self.linear_1.backward_update_gradient(X, delta_tan)
            self.linear_2.backward_update_gradient(res2, delta_tan2)
            self.linear_3.backward_update_gradient(res4, delta_soft)

            # mise a jour  de la matrice de poids
            self.linear_1.update_parameters(learning_rate)
            self.linear_2.update_parameters(learning_rate)
            self.linear_3.update_parameters(learning_rate)

            self.linear_1.zero_grad()
            self.linear_2.zero_grad()
            self.linear_3.zero_grad()

    def predict(self,xtest,biais = True):
        if biais == True:
            bias = np.ones((len(xtest), 1))
            xtest = np.hstack((bias, xtest))
        
        res1 = self.linear_1.forward(xtest)
        res2 = self.tanh.forward(res1)
        res3 = self.linear_2.forward(res2)
        res4 = self.tanh2.forward(res3)
        res5 = self.linear_3.forward(res4)
        res6 = self.softmax.forward(res5)

        # return np.where(res4>=0.5,1,0)
        return np.argmax(res6, axis=1)
"""
uspsdatatrain = "../data/USPS_train.txt"
uspsdatatest = "../data/USPS_test.txt"
alltrainx, alltrainy = load_usps(uspsdatatrain)
alltestx, alltesty = load_usps(uspsdatatest)

# taille couche
alltrainy_oneHot = OneHotEncoding(alltrainy)

if alltrainy.ndim == 1 : 
    alltrainy = alltrainy.reshape((-1,1))

nombre_neurone = 10
soft2 = soft2()
soft2.fit(alltrainx,alltrainy_oneHot,nombre_neurone=nombre_neurone,n_iter=100,learning_rate=0.001)

# affichage de la courbe d'errreur
plt.figure()
plt.title('erreur en fonction de literation')
plt.plot(soft2.list_error, label='loss')
plt.legend()
plt.xlabel('itérations')
plt.ylabel('erreur')
plt.show()
"""
##########################################################################################################
##########################################################################################################
##########################################################################################################

# TEST PARTIE 5
