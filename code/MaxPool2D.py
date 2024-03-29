"""
Classe ABSTRAITE MaxPool2D.
ABITBOL YOSSEF
DUFOURMANTELLE JEREMY
"""

import numpy as np

from projet_etu import Module

class MaxPool2D(Module):

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
        size_h = ((X.shape[1] - self._k_size) // self._stride) + 1
        size_w = ((X.shape[2] - self._k_size) // self._stride) + 1
        outPut=np.zeros((X.shape[0],size_h,size_w,X.shape[-1]))
        for i in range(0, size_h, self._stride):
            for j in range(0, size_w, self._stride):
                outPut[:, i,j, :] = np.max(X[:,i:i+self._k_size, j:j + self._k_size, :], axis=(1,2))
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
        size_h = ((input.shape[1] - self._k_size) // self._stride) + 1
        size_w = ((input.shape[2] - self._k_size) // self._stride) + 1
        batch=input.shape[0]
        chan_in=input.shape[3]
        outPut=np.zeros((input.shape[0],input.shape[1]*input.shape[2],input.shape[3]))
        for i in range(0, size_h, self._stride):
            for j in range(0, size_w, self._stride):
                region_shaped=input[:, i:i+self._k_size,j:j+self._k_size,:].reshape(-1,self._k_size*self._k_size,chan_in)
                indexes_argmax=np.argmax(region_shaped, axis=1) + i + j
                outPut[np.repeat(range(batch),chan_in),indexes_argmax.flatten(),list(range(chan_in))*batch]=delta[:,i,j,:].reshape(-1)
        self._delta=outPut.reshape(input.shape)
        return self._delta