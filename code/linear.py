import numpy as np
from  projet_etu import Module

class Linear(Module):
    """Implementation du module Lineaire """
    def __init__(self,input,output):
        self.input = input 
        self.output = output 
        self._parameters = np.random.rand(self.input, self.output) 
        self._gradient = np.zeros((input,output))
        
    def zero_grad(self):
        """Annule gradient"""
        self._gradient = np.zeros(self._gradient.shape) 

    def forward(self, X):
        """Calcule la passe forward, calcul des sorties en fonctions des entrees X"""
        return np.dot(X, self._parameters)

    def update_parameters(self, gradient_step=1e-3):
        """Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step"""
        self._parameters = self._parameters - gradient_step*self._gradient 

    def backward_update_gradient(self, input, delta):
        """Met a jour la valeur du gradient"""
        print("backward_update_gradient")
        print("taille input:",input.shape," taille delta:",delta.shape)
        self._gradient = self._gradient + np.dot( input.T, delta ) 

    def backward_delta(self, input, delta):
        """Calcul la derivee de l'erreur"""
        print("backward_delta")
        print("delta shape:",delta.shape,"input shape:",input.shape,"parameter",self._parameters.shape)
        return np.dot(delta,self._parameters.T) 
