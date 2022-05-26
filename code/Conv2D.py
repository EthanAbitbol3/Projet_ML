"""
Classe ABSTRAITE Conv2D.
ABITBOL YOSSEF
DUFOURMANTELLE JEREMY
"""

import numpy as np

from projet_etu import Module

class Conv2D(Module):

    def __init__(self, k_size, chan_in, chan_out, stride=1, biais=True):
        """Initialisation des parametres"""
        self.k_size=k_size
        self.chan_in=chan_in
        self.chan_out=chan_out
        self.stride=stride
        facteur=1 / np.sqrt(chan_in*k_size)
        self._parameters = np.random.uniform(-facteur, facteur, (k_size,k_size,chan_in,chan_out))
        self._gradient=np.zeros(self._parameters.shape)
        self.biais = biais
        if(self.biais):
            self._biais=np.random.uniform(-facteur, facteur, chan_out)
            self._gradbiais = np.zeros((chan_out))

    def zero_grad(self):
        """Annule gradient"""
        self._gradient = np.zeros(self._gradient.shape)
        if (self.biais):
            self._gradbiais = np.zeros(self._gradbiais.shape)

    def forward(self, X):
        """Calcule la passe forward"""
        size_h = ((X.shape[1] - self.k_size) // self.stride) + 1
        size_w = ((X.shape[2] - self.k_size) // self.stride) + 1
        outPut=np.zeros((X.shape[0],size_h,size_w,self.chan_out))
        for i in range(0,size_h,self.stride):
            for j in range(0,size_w,self.stride):
                outPut[:,i,j,:]=X[:,i: i + self.k_size,j: j + self.k_size,:].reshape(X.shape[0],-1) @ self._parameters.reshape(-1,self.chan_out)
        if (self.biais):
            outPut += self._biais
        self._forward = outPut
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
        """Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step"""
        self._parameters -= gradient_step * self._gradient
        if self.biais:
            self._biais -= gradient_step * self._gradbiais

    def backward_update_gradient(self, input, delta):
        """Met a jour la valeur du gradient"""
        size_h = ((input.shape[1] - self.k_size) // self.stride) + 1
        size_w = ((input.shape[2] - self.k_size) // self.stride) + 1
        outPut=np.zeros(self._gradient.shape)
        for i in range(0, size_h, self.stride):
            for j in range(0, size_w, self.stride):
                    res=(delta[:, i,j, :].T) @ (input[:, i: i + self.k_size,j: j + self.k_size, :].reshape(input.shape[0], -1))
                    outPut+=res.reshape(res.shape[0],self.k_size,self.k_size,input.shape[-1]).transpose(1,2,3,0)
        self._gradient=outPut/delta.shape[0]
        if self.biais:
            self._gradbiais=delta.mean((0,1,2))

    def backward_delta(self, input, delta):
        """Calcul la derivee de l'erreur"""
        size_h = ((input.shape[1] - self.k_size) // self.stride) + 1
        size_w = ((input.shape[2] - self.k_size) // self.stride) + 1
        outPut = np.zeros(input.shape)
        for i in range(0, size_h, self.stride):
            for j in range(0, size_w, self.stride):
                res=(delta[:,i,j,:])@(self._parameters.reshape(-1,self.chan_out).T)
                outPut[:, i:i + self.k_size, j:j + self.k_size, :]+=res.reshape(delta.shape[0],self.k_size,self.k_size,self.chan_in)
        self._delta=outPut
        return self._delta