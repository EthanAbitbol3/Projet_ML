from projet_etu import Loss
import numpy as np

class MSELoss(Loss):
    """implementation de la classe Mean square loss"""
    def forward(self, y, yhat):
        return np.linalg.norm( y.reshape((-1,1))-yhat.reshape((-1,1)), axis=1) ** 2

    def backward(self, y, yhat):
        print("y shape: ",y.shape,"yhat shape : ",yhat.shape)
        return -2 * (y.reshape((-1,1))-yhat.reshape((-1,1)))

