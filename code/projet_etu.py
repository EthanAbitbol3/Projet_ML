"""
Classe ABSTRAITE Module qui représente un module générique du réseau de neurones.


"""
import numpy as np

class Loss(object):
    def forward(self, y, yhat):
        """Calculer le cout en fonction de deux entrees"""
        pass

    def backward(self, y, yhat):
        """calcul le gradient du cout par rapport yhat"""
        pass


class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        """Annule gradient"""
        pass

    def forward(self, X):
        """Calcule la passe forward"""
        pass

    def update_parameters(self, gradient_step=1e-3):
        """Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step"""
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        """Met a jour la valeur du gradient"""
        pass

    def backward_delta(self, input, delta):
        """Calcul la derivee de l'erreur"""
        pass
