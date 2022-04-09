from linear import Linear
from MSELoss import MSELoss
from TanH import Tanh
from Sigmoide import Sigmoide
from Sequentiel import Sequentiel
from Optim import Optim, SGD
from sklearn.datasets import make_blobs,make_moons,make_regression
from matplotlib import pyplot as plt
import numpy as np 
from mltools import plot_data, plot_frontiere, make_grid, gen_arti
from sklearn.preprocessing import StandardScaler

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

class neural_network_non_lineaire:
    def __init__(self, list_error):
        self.list_error = list_error

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

        self.list_errors = []

        for _ in range(n_iter):
            
            # phase forward
            res1 = self.linear_1.forward(X)
            res2 = self.tanh.forward(res1)
            res3 = self.linear_2.forward(res2)
            res4 = self.sigmoide.forward(res3)

            self.list_errors.append(np.sum(self.mse.forward(y, res4))) # loss

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

        return np.argmax(res4, axis = 1) 

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
neural_network_non_lineaire = neural_network_non_lineaire([])
neural_network_non_lineaire.fit(X,y,nombre_neurone=nombre_neurone,n_iter=100,learning_rate=0.01)
y = np.argmax(y,axis=1)

# affichage de la frontiere de decision ainsi que des donnees
plt.figure()
plot_frontiere(X,lambda x : neural_network_non_lineaire.predict(x),step=100)
plot_data(X,y.reshape(1,-1)[0])
plt.title(f"neural network non lineaire avec {nombre_neurone} neurones")
plt.show()

# affichage de la courbe d'errreur
plt.figure()
plt.title('erreur en fonction de literation')
plt.plot(neural_network_non_lineaire.list_errors, label='loss')
plt.legend()
plt.xlabel('itérations')
plt.ylabel('erreur')
plt.show()