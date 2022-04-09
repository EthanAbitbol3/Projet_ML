"""
Classe ABSTRAITE Sigmoide.
ABITBOL YOSSEF
DUFOURMANTELLE JEREMY
"""

from projet_etu import Module
import numpy as np

class Sigmoide(Module):
    """ Implementation du module sigmoide"""
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        """Annule gradient"""
        pass

    def forward(self, X):
        """Calcule la passe forward"""
        return (1 / (1 + np.exp(-X)))

    def update_parameters(self, gradient_step=1e-3):
        """Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step"""
        pass

    def backward_update_gradient(self, input, delta):
        """Met a jour la valeur du gradient"""
        pass

    def backward_delta(self, input, delta):
        """Calcul la derivee de l'erreur"""
        return ( np.exp(-input) / ( 1 + np.exp(-input) )**2 ) * delta 
