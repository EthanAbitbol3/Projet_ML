from linear import Linear
from MSELoss import MSELoss
from TanH import Tanh
from Sigmoide import Sigmoide
from sklearn.datasets import make_blobs,make_moons,make_regression
import matplotlib.pyplot as plt

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

def neural_network_lineaire(X_train, y_train, nombre_neurone , n_iter = 100,learning_rate = 0.001):
    # nombre_neurone = y_train.shape[1]
    modele = Linear(X_train.shape[1],nombre_neurone)
    loss = MSELoss()
    train_loss = []

    for _ in range(n_iter):
        # Calcul de la phase forward
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

def neural_network_lineaire_bis(X_train, y_train, nombre_neurone , n_iter = 100,learning_rate = 0.001):
    # nombre_neurone = y_train.shape[1]
    module1 = Linear(X_train.shape[1],nombre_neurone)
    module2 = Linear(nombre_neurone,y.shape[1])
    loss = MSELoss()
    train_loss = []

    for _ in range(n_iter):
        # Calcul de la phase forward
        y1 = module1.forward(X_train)
        y_hat = module2.forward(y1)

        print(module1._parameters)
        # Calcul de l'erreur
        forward_loss = loss.forward(y_train,y_hat)

        # Back propagation
        backward_loss = loss.backward(y_train,y_hat)
        delta_module2 = module2.backward_delta(y1,backward_loss)

        # Mise a jour du gradient
        module2.backward_update_gradient(y1,backward_loss)
        module1.backward_update_gradient(X_train,delta_module2)

        train_loss.append(forward_loss.sum()) # erreur
        
        # mise a jour  de la matrice de poids 
        module1.update_parameters(learning_rate)
        module2.update_parameters(learning_rate)
                
        module1.zero_grad() 
        module2.zero_grad()
    return module2, train_loss

# Génération des points
nombre_dim_y = 1
X, y = make_regression(n_samples=100, n_features=1,bias=0.5,noise=10,n_targets=nombre_dim_y, random_state=0)
if y.ndim == 1 : 
    y = y.reshape((-1,1))
print(y.shape)
modele, train_loss = neural_network_lineaire(X,y,nombre_neurone=1)
# affichage(X,y,modele,train_loss)

modele2, train_loss2 = neural_network_lineaire_bis(X, y, 5 , n_iter = 100,learning_rate = 0.001)
print(modele2._parameters)
affichage(X,y,modele2,train_loss2)