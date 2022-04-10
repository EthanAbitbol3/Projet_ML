"""
Classe ABSTRAITE MSELoss.
ABITBOL YOSSEF
DUFOURMANTELLE JEREMY
"""

from projet_etu import Loss
import numpy as np

class MSELoss(Loss):
    """implementation de la classe Mean square loss"""
    def forward(self, y, yhat):
        """||y-y^||^2"""
        # assert (y.shape == yhat.shape)
        return np.linalg.norm( y - yhat, axis=1) ** 2

    def backward(self, y, yhat):
        """derive de la forward"""
        # assert (y.shape == yhat.shape)
        return -2 * (y-yhat)

