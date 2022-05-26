"""
Classe ABSTRAITE Softmax.
ABITBOL YOSSEF
DUFOURMANTELLE JEREMY
"""

import numpy as np

from projet_etu import Module


class Softmax(Module):
    def __init__(self):
        """Initialisation des parametres"""
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        """Annule gradient"""
        pass

    def forward(self, X):
        """Calcule la passe forward"""
        sum_exp = np.sum(np.exp(X), axis=1,keepdims=True)
        self._forward = np.exp(X)/sum_exp
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
        """Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step"""
        pass

    def backward_update_gradient(self, input, delta):
        """Met a jour la valeur du gradient"""
        pass

    def backward_delta(self, input, delta):
        """Calcul la derivee de l'erreur"""
        sum_exp = np.sum(np.exp(input), axis=1,keepdims=True)
        z = np.exp(input)/sum_exp
        derive = z * (1 - z)
        self._delta = delta * derive
        return self._delta