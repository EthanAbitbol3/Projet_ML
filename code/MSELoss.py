from projet_etu import Loss
import numpy as np

class MSELoss(Loss):
    """implementation de la classe Mean square loss"""
    def forward(self, y, yhat):
        return np.linalg.norm( y-yhat, axis=1) ** 2

    def backward(self, y, yhat):
        return -2 * (y-yhat)

