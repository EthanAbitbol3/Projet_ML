"""
Classe ABSTRAITE Conv1D.
ABITBOL YOSSEF
DUFOURMANTELLE JEREMY
"""

from  projet_etu import Module

import numpy as np

class Conv1D(Module):

    def __init__(self, k_size, chan_in, chan_out, stride=1, biais=True):
        """Initialisation des parametres"""
        self.k_size=k_size
        self.chan_in=chan_in
        self.chan_out=chan_out
        self.stride=stride
        facteur=1 / np.sqrt(chan_in*k_size)
        self._parameters = np.random.uniform(-facteur, facteur, (k_size,chan_in,chan_out))
        self._gradient=np.zeros(self._parameters.shape)
        self.biais = biais
        if(self.biais):
            self._biais=np.random.uniform(-facteur, facteur, chan_out)
            self._gradbiais = np.zeros((chan_out))

    def zero_grad(self):
        """Annule gradient"""
        self._gradient=np.zeros(self._gradient.shape)
        if (self.biais):
            self._gradbiais = np.zeros(self._gradbiais.shape)

    def forward(self, X):
        """Calcule la passe forward"""
        size = ((X.shape[1] - self.k_size) // self.stride) + 1
        output=np.array([(X[:, i: i + self.k_size, :].reshape(X.shape[0], -1)) @ (self._parameters.reshape(-1, self.chan_out)) \
                         for i in range(0,size,self.stride)])
        if (self.biais):
            output+=self._biais
        self._forward=output.transpose(1,0,2)
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
        """Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step"""
        self._parameters -= gradient_step * self._gradient
        if self.biais:
            self._biais -= gradient_step * self._gradbiais

    def backward_update_gradient(self, input, delta):
        """Met a jour la valeur du gradient"""
        size = ((input.shape[1] - self.k_size) // self.stride) + 1
        output = np.array([ (delta[:,i,:].T) @ (input[:, i: i + self.k_size, :].reshape(input.shape[0], -1))  \
                           for i in range(0, size, self.stride)])
        self._gradient=np.sum(output,axis=0).T.reshape(self._gradient.shape)/delta.shape[0]

        if self.biais:
            self._gradbiais=delta.mean((0,1))

    def backward_delta(self, input, delta):
        """Calcul la derivee de l'erreur"""
        outPut = np.zeros(input.shape)
        for i in range(0, (((input.shape[1] - self.k_size) // self.stride) + 1), self.stride):
            outPut[:,i:i+self.k_size,:] += ((delta[:, i, :]) @ (self._parameters.reshape(-1,self.chan_out).T)).reshape(input.shape[0],self.k_size,self.chan_in)
        self._delta= outPut
        return self._delta