"""
Classe ABSTRAITE ReLU.
ABITBOL YOSSEF
DUFOURMANTELLE JEREMY
"""

from projet_etu import Module

import numpy as np

class ReLU(Module):

    def __init__(self , threshold):
        """Initialisation des parametres"""
        self._parameters = None
        self._gradient = None
        self._threshold = threshold

    def zero_grad(self):
        """Annule gradient"""
        pass

    def forward(self, X):
        """Calcule la passe forward"""
        self._forward =  np.where(X>self._threshold,X,0.)
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
        """Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step"""
        pass

    def backward_update_gradient(self, input, delta):
        """Met a jour la valeur du gradient"""
        pass

    def backward_delta(self, input, delta):
        """Calcul la derivee de l'erreur"""
        derive = (input > self._threshold).astype(float)
        self._delta= delta * derive
        return self._delta