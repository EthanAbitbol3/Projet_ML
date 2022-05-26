"""
Classe ABSTRAITE BCELoss.
ABITBOL YOSSEF
DUFOURMANTELLE JEREMY
"""

from projet_etu import Loss

import numpy as np

class BCELoss(Loss):
    
    def forward(self, y, yhat,eps = 1e-100):
        """Calculer le cout en fonction de deux entrees"""
        return - (y * np.log(yhat + eps) + (1 - y) * np.log(1 - yhat + eps))
    
    def backward(self, y, yhat,eps = 1e-100):
        """calcul le gradient du cout par rapport yhat"""
        return ((1 - y) / (1 - yhat + eps)) - (y / (yhat + eps))