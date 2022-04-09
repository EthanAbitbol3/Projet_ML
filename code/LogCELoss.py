from projet_etu import Loss
import numpy as np

class LogCELoss(Loss):
    """implementation de la classe Mean square loss"""
    def forward(self, y, yhat):
        assert (y.shape == yhat.shape)
        return np.sum(- y * yhat)

    def backward(self, y, yhat):
        assert (y.shape == yhat.shape)
        return -y

