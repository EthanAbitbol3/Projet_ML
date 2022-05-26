"""
Classe ABSTRAITE AvgPool1D.
ABITBOL YOSSEF
DUFOURMANTELLE JEREMY
"""

import numpy as np

from projet_etu import Module


class AvgPool1D(Module):

    def __init__(self, k_size=3, stride=1):
        """Initialisation des parametres"""
        self._parameters = None
        self._gradient = None
        self._k_size = k_size
        self._stride = stride

    def zero_grad(self):
        """Annule gradient"""
        pass

    def forward(self, X):
        """Calcule la passe forward"""
        outPut = np.zeros((X.shape[0], (((X.shape[1] - self._k_size) // self._stride) + 1), X.shape[2]))
        for i in range(0, (((X.shape[1] - self._k_size) // self._stride) + 1), self._stride):
            outPut[:,i,:]=np.mean(X[:,i:i+self._k_size,:],axis=1)
        self._forward=outPut
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
        """Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step"""
        pass

    def backward_update_gradient(self, input, delta):
        """Met a jour la valeur du gradient"""
        pass

    def backward_delta(self, input, delta):
        """Calcul la derivee de l'erreur"""
        outPut = np.zeros(input.shape)
        for i in range(0,(((input.shape[1] - self._k_size) // self._stride) + 1),self._stride):
            res=np.ones((self._k_size,input.shape[0], input.shape[2])) * delta[:, i, :] / self._k_size
            outPut[:,i:i+self._k_size,:]=res.transpose(1,0,2)
        self._delta=outPut
        return self._delta