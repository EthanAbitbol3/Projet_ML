"""
Classe ABSTRAITE MSELoss.
ABITBOL YOSSEF
DUFOURMANTELLE JEREMY
"""

from projet_etu import Loss
import numpy as np

class MSELoss(Loss):

    def forward(self, y, yhat):
        """Calculer le cout en fonction de deux entrees"""
        return np.sum((y-yhat)**2,axis=1,keepdims=True)

    def backward(self, y, yhat):
        """calcul le gradient du cout par rapport yhat"""
        return 2*(yhat-y)

