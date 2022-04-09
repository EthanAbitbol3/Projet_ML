"""
Classe ABSTRAITE Sequentiel.
ABITBOL YOSSEF
DUFOURMANTELLE JEREMY
"""

from sys import modules
import numpy as np 

class Sequentiel():
    """Implementation de la classe Sequentiel"""
    def __init__(self,modules,loss) -> None:
        self.modules = modules # liste des differents modules
        self.loss = loss # fonctions loss
        self.list_error = None # liste des erreurs
        self.res = [] # liste des forward
        self.delta = [] # liste des deltas

    def zero_grad(self):
        """Annule gradient"""
        for module in self.modules:
            module.zero_grad()

    # def forward(self, X):
    #     """Calcule la passe forward, calcul des sorties en fonctions des entrees X"""
    #     self.res.append(self.modules[0].forward(X))
    #     for i in range (1,len(self.modules)):
    #         self.res.append(self.modules[i].forward(self.res[-1])) # on utilise le res du forward precedent 

    def update_parameters(self, gradient_step=1e-3):    
        """Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step"""
        for module in self.modules:
            module.update_parameters(gradient_step)

    def backward_update_gradient(self, input, delta):
        """Met a jour la valeur du gradient"""
        modules_inverse = self.modules[::-1]
        next = modules_inverse[0]
        cpt = len(self.modules)-2
        i = 0
        for module in modules_inverse[1:]:
            previous = module
            next.backward_update_gradient(self.res[cpt],delta)
            delta = self.delta[i]
            next = previous 
            i+=1
            cpt-=1
        next.backward_update_gradient(input,delta)

    # def backward_delta(self, input, delta):
    #     """Calcul la derivee de l'erreur"""
    #     modules_inverse = self.modules[::-1]
    #     next = modules_inverse[0]
    #     cpt = len(self.modules)-2
    #     for module in modules_inverse[1:]:
    #         previous = module 
    #         self.delta.append(next.backward_delta(self.res[cpt],delta))
    #         delta = ...
    #         next = previous
    #     self.delta.append(next.backward_delta(input,delta))
    #     return self.delta


    def fit(self,X, y):
        # partie forward 
        self.res.append(self.modules[0].forward(X))
        for i in range (1,len(self.modules)):
            self.res.append(self.modules[i].forward(self.res[-1])) # on utilise le res du forward precedent 

        self.error = np.sum(self.loss.forward(y, self.res[-1]))

        # partie backward 
        self.delta.append(self.loss.backward( y, self.res[-1] ))
        modules_inverse = self.modules[::-1]
        res_inverse = self.res[::-1]
        for i in range(0,len(modules_inverse)-1): # on parcours 
            self.delta.append(modules_inverse[i].backward_delta( res_inverse[i+1], self.delta[-1] ) )
        

    def predict(self,xtest,biais = True):
        if biais == True:
            bias = np.ones((len(xtest), 1))
            xtest = np.hstack((bias, xtest))

        res_pred = []
        res_pred.append(self.modules[0].forward(xtest))
        for i in range (1,len(self.modules)):
            res_pred.append(self.modules[i].forward(res_pred[-1])) # on utilise le res du forward precedent 
        
        return np.argmax(res_pred[-1], axis = 1) 