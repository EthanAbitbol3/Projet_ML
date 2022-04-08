import numpy as np
from Sequentiel import Sequentiel

class Optim():
    """implementation de la classe Optim"""
    def __init__(self,net,loss,eps) -> None:
        self.net = net # resequ net 
        self.loss = loss # fonction de cout 
        self.eps = eps  # le pas 
        self.error = None
        self.sequentiel = Sequentiel(net,loss)
        
    def step(self, batch_x, batch_y):
        """
        1 etape de la descente de gradient 
        """
        self.sequentiel.fit(batch_x, batch_y)
        self.error = self.sequentiel.error
        # mise a jour backward_update_gradient
        for i in range(len(self.net)):

            if i == 0:
                self.net[i].backward_update_gradient(batch_x, self.sequentiel.delta[-1])
            else:
                self.net[i].backward_update_gradient(self.sequentiel.res[i-1], self.sequentiel.delta[-i-1])

            # mise a jour parametres
            self.net[i].update_parameters(self.eps)
            self.net[i].zero_grad()

    def predict(self,xtest, biais = True):
        if biais == True:
            bias = np.ones((len(xtest), 1))
            xtest = np.hstack((bias, xtest))

        res_pred = []
        res_pred.append(self.net[0].forward(xtest))
        for i in range (1,len(self.net)):
            res_pred.append(self.net[i].forward(res_pred[-1])) # on utilise le res du forward precedent 
        
        return np.argmax(res_pred[-1], axis = 1) 

class SGD():
    def __init__(self,net, loss, batch_x, batch_y) -> None:
        self.net = net
        self.loss = loss
        self.batch_x = batch_x
        self.batch_y = batch_y
        self.list_error = []

    def fit(self, taille_batch=1, n_iter= 1000, biais= True, learning_rate = 0.001):
        if biais == True:
            bias = np.ones((len(self.batch_x), 1))
            self.batch_x = np.hstack((bias, self.batch_x))

        optim  = Optim(self.net, self.loss, learning_rate)

        card = self.batch_x.shape[0]
        nb_batchs = card//taille_batch
        inds = []
        for i in range(card):
            inds.append(i)

        # Création des batchs
        np.random.shuffle(inds)
        # print(nb_batchs)
        # for i in range(nb_batchs):
        #     for j in inds[i*taille_batch:(i+1)*taille_batch]:
        #         batchs = [j]
        batchs = [[j for j in inds[i*taille_batch:(i+1)*taille_batch]] for i in range(nb_batchs)]

        for i in range(n_iter):
            # On mélange de nouveau lorsqu'on a parcouru tous les batchs
            if i%nb_batchs == 0:
                np.random.shuffle(inds)
                batchs = [[j for j in inds[i*taille_batch:(i+1)*taille_batch]] for i in range(nb_batchs)]

            # Mise-à-jour sur un batch
            batch = batchs[i%(nb_batchs)]
            optim.step(self.batch_x[batch], self.batch_y[batch])
            self.list_error.append(optim.error)
            

    def predict(self, xtest, biais=True):
        """ Prédiction sur des données. Il s'agit simplement d'un forward.
        """
        if biais == True:
            bias = np.ones((len(xtest), 1))
            xtest = np.hstack((bias, xtest))

        res_pred = []
        res_pred.append(self.net[0].forward(xtest))
        for i in range (1,len(self.net)):
            res_pred.append(self.net[i].forward(res_pred[-1])) # on utilise le res du forward precedent 
        
        return np.argmax(res_pred[-1], axis = 1)
