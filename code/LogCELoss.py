"""
Classe ABSTRAITE LogCELoss.
ABITBOL YOSSEF
DUFOURMANTELLE JEREMY
"""

from projet_etu import Loss
import numpy as np
from Softmax import Softmax
from LogSoftmax import LogSoftmax

class LogCELoss(Loss):
    """implementation de la classe Mean square loss"""
    def forward(self, y, yhat):
        y = y.reshape(-1,1).T
        assert (y.shape == yhat.shape)
        return np.sum(-y*yhat,axis=1)+np.log(np.sum(np.exp(yhat),axis=1))

    def backward(self, y, yhat):
        y = y.reshape(-1,1).T
        print("yshape",y.shape,"yhat :",yhat.shape)
        assert (y.shape == yhat.shape)
        s = LogSoftmax().forward( yhat )
        return - y + s * ( 1 - s )

