import numpy as np
from projet_etu import Module

class LogSoftmax(Module):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        """Annule gradient"""
        pass

    def forward(self, X):
        """Calcule la passe forward"""
        logsum_exp = np.log(np.sum(np.exp(X), axis=1))
        return X - logsum_exp

    def update_parameters(self, gradient_step=1e-3):
        """Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step"""
        pass 

    def backward_update_gradient(self, input, delta):
        """Met a jour la valeur du gradient"""
        pass

    def backward_delta(self, input, delta):
        """Calcul la derivee de l'erreur"""
        sum_exp = np.sum(np.exp(input), axis=1)
        return delta * (np.exp(input)/sum_exp) * (1 - (np.exp(input)/sum_exp))
