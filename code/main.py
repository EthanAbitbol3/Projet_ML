from linear import Linear
from MSELoss import MSELoss
from TanH import Tanh
from Sigmoide import Sigmoide
import Sequentiel,Optim
from sklearn.datasets import make_blobs,make_moons,make_regression
from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd 
from mltools import plot_data, plot_frontiere, make_grid, gen_arti


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
        backward_loss = loss.backward(y_train,y_hat)
        modele.backward_update_gradient(X_train,backward_loss)

        train_loss.append(forward_loss.sum()) # erreur

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
# nombre_dim_y = 2
# X, y = make_regression(n_samples=100, n_features=1,bias=0.5,noise=10,n_targets=nombre_dim_y, random_state=0)
# if y.ndim == 1 : 
#     y = y.reshape((-1,1))
# modele, train_loss = neural_network_lineaire(X,y,nombre_neurone= y.shape[1])
# affichage(X,y,modele,train_loss)

##########################################################################################################
##########################################################################################################
##########################################################################################################

# TEST PARTIE 2: NON LINEAIRE MODULE

def neural_network_non_lineaire(X, y, nombre_neurone, n_iter = 100 , learning_rate = 0.001, biais = True):

    if biais == True:
        bias = np.ones((len(X), 1))
        X = np.hstack((bias, X))

    # Initialisation des modules
    mse = MSELoss()
    linear_1 = Linear(X.shape[1], nombre_neurone)
    tanh = Tanh()
    linear_2 = Linear(nombre_neurone, X.shape[1])
    sigmoide = Sigmoide()

    list_errors = []

    for _ in range(n_iter):
        
        # phase forward
        res1 = linear_1.forward(X)
        res2 = tanh.forward(res1)
        res3 = linear_2.forward(res2)
        res4 = sigmoide.forward(res3)

        list_errors.append(np.mean(mse.forward(y, res4))) # loss

        #  retro propagation du gradient de la loss par rapport aux parametres et aux entrees
        last_delta = mse.backward(y, res4)

        delta_sig = sigmoide.backward_delta(res3, last_delta)
        delta_lin = linear_2.backward_delta(res2, delta_sig)
        delta_tan = tanh.backward_delta(res1, delta_lin)

        linear_1.backward_update_gradient(X, delta_tan)
        linear_2.backward_update_gradient(res2, delta_sig)

        # mise a jour  de la matrice de poids
        linear_1.update_parameters(learning_rate)
        linear_2.update_parameters(learning_rate)

        linear_1.zero_grad()
        linear_2.zero_grad()

    return list_errors,linear_1,linear_2, res4

# generations de points 
X, y = gen_arti(data_type=1)
nombre_neurone = 2
_,_,_, res = neural_network_non_lineaire(X, y, nombre_neurone, n_iter = 100 , learning_rate = 0.001, biais = True)
def f(res):
    return  np.argmax(res, axis = 1)
plt.figure()
plot_frontiere(X,f,step=100)
plot_data(X,y.reshape(1,-1)[0])
plt.title(f"neural network non lineaire avec {nombre_neurone} neurones")
plt.show()



# X2, y2 = make_moons(n_samples=100, noise=0.2, random_state = 0 )
# plt.figure(1)
# df = pd.DataFrame(dict(x=X2[:,0], y=X2[:,1], label=y2))
# colors = {0:'red', 1:'blue'}
# fig, ax = plt.subplots()
# grouped = df.groupby('label')
# for key, group in grouped:
#     group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
# plt.show()


# # print("avant ",y2.shape)
# y2 = y2.reshape((-1,1))
# # print("apres ",y2.shape)

# ll = []
# ll.append(neural_network_non_lineaire(X2, y2, 2))
# ll.append(neural_network_non_lineaire(X2, y2, 3))
# ll.append(neural_network_non_lineaire(X2, y2, 20))

# for i in range(len(ll)):
#     title = "erreur du réseau"
#     plt.figure(i)
#     plt.plot(np.arange(len(ll[i][0])),ll[i][0])
#     plt.title(title)
#     plt.show()

# plt.figure(5)
# for key, group in grouped:
#     group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
# plt.plot(X2[:,0],X2[:,0] *ll[0][1]._parameters[0][0])
# plt.show()

# datax, datay = gen_arti(data_type = 1 , epsilon = 0.01)
# datay = np.array([ 0 if d == -1 else 1 for d in datay ])
# datay = datay.reshape((-1,1))

# grid, x_grid, y_grid = make_grid(xmin=-2, xmax=2, ymin=-2, ymax=2, step=1000)
# list_errors,linear_1,linear_2 = neural_network_non_lineaire(datax, datay, 2, n_iter=1000)

# datay = np.argmax(datay, axis=1)
# linear_3 = Linear(datax.shape[1], 2)
# tanh_3 = Tanh()
# linear_4 = Linear(2, datay.shape[1])
# sigmoide_4 = Sigmoide()

# def add_bias(datax):
#     """ Fonction permettant d'ajouter un biais aux données.
#         @param xtrain: float array x array, données auxquelle ajouter un biais
#     """
#     bias = np.ones((len(datax), 1))
#     return np.hstack((bias, datax))

# def predict(x):
#     x = add_bias (x)
#     res = linear_3.forward(datax)
#     res = tanh_3.forward(res)
#     res = linear_4.forward(res)
#     res = sigmoide_4.forward(res)
#     return np.argmax(res, axis = 1)

# frontiere de decision
# plt.figure()
# plt.title('frontiére de décision de bruit ')
# plot_frontiere(datax,lambda x : np.sign(x.dot(linear_2._parameters)),step = 100)
# plot_data(datax,datay)
# plt.show()

# courbe d'erreur 
# plt.figure()
# plt.plot(np.arange(len(list_errors)),list_errors)
# plt.show()

## Visualisation de la fonction de coût en 2D
# plt.figure()
# plt.title('Visualisation de la fonction de coût en 2D')
# plt.contourf(x_grid,y_grid,np.array([mse(w,datax,datay).mean() for w in grid]).reshape(x_grid.shape),levels=20)
# plt.scatter( [w[0] for w in list_w] , [w[1] for w in list_w], c='cyan', marker='*')

# X , y = gen_arti(data_type=1)
# list_errors,linear_1,linear_2,res4 = neural_network_non_lineaire(X, y, 2, n_iter=1000,learning_rate=0.01)

# def f(X):
    # _ , y = gen_arti(data_type=1)
    # # y = y.reshape((1,y.shape[0]))
    # _,_,_,res4 = neural_network_non_lineaire(X, y.T, 2, n_iter=1000,learning_rate=0.01)
    # return np.where(X<0.5, 0, 1)

# plt.figure()
# plot_data(X,y)
# plot_frontiere(X,f)
# plt.show()