"""
Classe Sequentiel
ABITBOL YOSSEF
DUFOURMANTELLE JEREMY
"""
import numpy as np
from projet_etu import Module 

class Sequentiel(Module) :

    def __init__(self,modules):
        super().__init__()
        self.modules = modules
        self.data = []
        self.deltas = []

    def zero_grad(self):
        """Annule gradient"""
        for module in self.modules :
            module.zero_grad()

    def forward(self, X):
        """Calcule la passe forward"""
        if len(self.modules) > 1 :
            self.data.append(self.modules[0].forward(X))
        else :
            return self.modules[0].forward(X)
        for i in range(1,len(self.modules)) :
            if i < len(self.modules) -1 :
                self.data.append(self.modules[i].forward(self.data[i-1]))
            else:
                return self.modules[i].forward(self.data[i-1])
        
    def update_parameters(self, gradient_step=1e-3):
        """Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step"""
        for module in self.modules:
            module.update_parameters(gradient_step)

    def backward_delta(self, input, delta):
        """Calcul la derivee de l'erreur"""
        data_reversed = self.data[::-1]
        module_reversed = self.modules[::-1]
        if len(self.modules) > 1 :
            self.deltas.append(module_reversed[0].backward_delta(data_reversed[0],delta))
        else :
            return module_reversed[0].backward_delta(input,delta)
        for i in range(1,len(module_reversed)):
            if i < len(self.modules)-1 :
                self.deltas.append(module_reversed[i].backward_delta(data_reversed[i],self.deltas[i-1]))
            else :
                return module_reversed[i].backward_delta(input,self.deltas[i-1])

    def backward_update_gradient(self, input,delta):
        """Met a jour la valeur du gradient"""
        deltas_reversed = self.deltas[::-1]
        deltas_reversed.append(delta)
        self.modules[0].backward_update_gradient(input, deltas_reversed[0])
        for i in range(1,len(self.modules)):
            self.modules[i].backward_update_gradient(self.data[i-1], deltas_reversed[i])

    def initialisation_parameters(self):
        """doit obligatoirement etre fait avant la passe forward !"""
        self.data = []
        self.deltas = []

    def predict(self,xtest,biais = True):
        if biais == True:
            bias = np.ones((len(xtest), 1))
            xtest = np.hstack((bias, xtest))
        self.initialisation_parameters()
        return np.where(self.forward(xtest)>=0.5,1,0)
