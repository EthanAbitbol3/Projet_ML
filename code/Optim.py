import numpy as np

from projet_etu import Loss, Module


class Optim():

    def __init__(self,net : Module,loss : Loss ,eps : float) -> None:
        self.net = net
        self.loss = loss
        self.eps = eps

    def step(self,batch_x,batch_y) -> float :
        """ Fonction qui calcule une passe d'apprentissage du réseau et retourne l'erreur courante"""

        # Calcul de la passe avant forward du réseau
        y_hat = self.net.forward(batch_x)

        # Calcul de la dérivé de la loss par rapport au y
        last_delta = self.loss.backward(batch_y, y_hat)
        
        # back propagation
        delta = self.net.backward_delta(batch_x, last_delta)

        # Mise a jour des gradients de chaque couche du réseau
        self.net.backward_update_gradient(batch_x,delta)

        # Mise a jour des parametres de chaque couche du réseau
        self.net.update_parameters()

        # Reinitialisation des gradients de chaque couche du réseau
        self.net.zero_grad()

        # Reinitialisation des parametres du reseau
        self.net.initialisation_parameters()

        return np.sum(self.loss.forward(batch_y, y_hat))

def split_data(batch_x , batch_y , batch_size : int) :
    number_split = len(batch_x) / batch_size
    return np.array_split(batch_x, number_split),np.array_split(batch_y, number_split)

def SGD(net : Module, loss : Loss,batch_x, batch_y , batch_size : int, nb_iter, eps : float):
    optim = Optim(net , loss , eps)
    array_batch_x,array_batch_y = split_data(batch_x, batch_y, batch_size)
    list_error = []
    for _ in range(nb_iter) : 
        index = np.random.randint(0,(len(array_batch_x)-1))
        list_error.append(optim.step(array_batch_x[index], array_batch_y[index]))
    return optim.net,list_error