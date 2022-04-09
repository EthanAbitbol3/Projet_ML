from projet_etu import Loss
import numpy as np

class CELoss(Loss):
    """implementation de la classe Mean square loss"""
    def forward(self, y, yhat):
        assert (y.shape == yhat.shape)
        return np.sum(-y*yhat,axis=1)+np.log(np.sum(np.exp(yhat),axis=1))

    def backward(self, y, yhat):
        assert (y.shape == yhat.shape)
        return -y + (y*np.exp(yhat))/np.sum(np.exp(yhat),axis=1)

