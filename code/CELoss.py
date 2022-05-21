"""
Classe ABSTRAITE CELoss.
ABITBOL YOSSEF
DUFOURMANTELLE JEREMY
"""

from projet_etu import Loss
import numpy as np

class CELoss(Loss):
    """implementation de la classe Mean square loss"""
    def forward(self, y, yhat):
        y = y.reshape(-1,1)
        #print("y.shape : ",y.shape," et yhat.shape = ",yhat.shape)
        assert (y.shape == yhat.shape)
        return np.sum(- y * yhat)

    def backward(self, y, yhat):
        y = y.reshape(-1,1)
        #print("y.shape : ",y.shape," et yhat.shape = ",yhat.shape)
        assert (y.shape == yhat.shape)
        
        return -y


