from projet_etu import Module
import numpy as np 

class Tanh(Module):
    """Implementation du module TanH"""
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        """Annule gradient"""
        pass

    def forward(self, X):
        """Calcule la passe forward"""
        return np.tanh(X)

    def update_parameters(self, gradient_step=1e-3):
        """Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step"""
        # self._parameters -= gradient_step*self._gradient
        pass

    def backward_update_gradient(self, input, delta):
        """Met a jour la valeur du gradient"""
        pass

    def backward_delta(self, input, delta):
        """Calcul la derivee de l'erreur"""
        return ( 1 - np.tanh(input)**2 ) * delta