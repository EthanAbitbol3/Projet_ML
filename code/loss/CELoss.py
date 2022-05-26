"""
Classe ABSTRAITE CELoss.
ABITBOL YOSSEF
DUFOURMANTELLE JEREMY
"""

from projet_etu import Loss

import numpy as np

class CELoss(Loss):

    def forward(self, y, yhat):
        """Calculer le cout en fonction de deux entrees"""
        return 1 - np.sum(yhat * y, axis = 1)

    def backward(self, y, yhat):
        """Calcul le gradient du cout par rapport yhat"""
        return yhat - y