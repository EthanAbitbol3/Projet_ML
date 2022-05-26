"""
Classe ABSTRAITE Linear.
ABITBOL YOSSEF
DUFOURMANTELLE JEREMY
"""

import numpy as np

from  projet_etu import Module

class Linear(Module):
    def __init__(self, input, output, biais=True):
        self.input=input
        self.output=output
        self._parameters =  2 * (np.random.rand(input, output) - 0.5)
        self._gradient = np.zeros((input, output))
        self.biais = biais
        if (self.biais):
            self._biais = 2 * (np.random.randn(output) - 0.5)
            self._gradbiais = np.zeros(output)

    def zero_grad(self):
        """Annule gradient"""
        self._gradient = np.zeros(self._gradient.shape)
        if (self.biais):
            self._gradbiais = np.zeros(self._gradbiais.shape)

    def forward(self, X):
        """Calcule la passe forward"""
        self._forward=np.dot(X,self._parameters)
        if(self.biais):
                self._forward = np.add(self._forward,self._biais)
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
        """Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step"""
        self._parameters -= gradient_step*self._gradient
        if(self.biais):
            self._biais -= gradient_step*self._gradbiais

    def backward_update_gradient(self, input, delta):
        """Met a jour la valeur du gradient"""
        self._gradient = np.dot(input.T, delta)
        if (self.biais):
            self._gradbiais = np.sum(delta,axis=0)

    def backward_delta(self, input, delta):
        """Calcul la derivee de l'erreur"""
        self._delta = np.dot(delta,self._parameters.T)
        return self._delta
        