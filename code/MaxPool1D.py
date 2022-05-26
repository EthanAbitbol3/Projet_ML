"""
Classe ABSTRAITE MaxPool1D.
ABITBOL YOSSEF
DUFOURMANTELLE JEREMY
"""

from projet_etu import Module
import numpy as np

class MaxPool1D(Module):

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
        size = ((X.shape[1] - self._k_size) // self._stride) + 1
        outPut = np.zeros((X.shape[0], size, X.shape[2]))
        for i in range(0, size, self._stride):
            outPut[:,i,:]=np.max(X[:,i:i+self._k_size,:],axis=1)
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
        size = ((input.shape[1] - self._k_size) // self._stride) + 1
        outPut=np.zeros(input.shape)
        batch=input.shape[0]
        chan_in=input.shape[2]
        for i in range(0,size,self._stride):
            indexes_argmax = np.argmax(input[:, i:i+self._k_size,:], axis=1) + i
            outPut[np.repeat(range(batch),chan_in),indexes_argmax.flatten(),list(range(chan_in))*batch]=delta[:,i,:].reshape(-1)
        self._delta=outPut
        return self._delta