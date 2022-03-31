"""
Classe ABSTRAITE Module qui représente un module générique du réseau de neurones.
"""


import numpy as np
from turtle import backward
from sklearn.model_selection import learning_curve
from sklearn.datasets import make_blobs,make_moons,make_regression
from matplotlib import pyplot as plt
import pandas as pd

#########################################################################################################
############################################## CLASSE LOSS ##############################################
#########################################################################################################

class Loss(object):
    def forward(self, y, yhat):
        """Calculer le cout en fonction de deux entrees"""
        pass

    def backward(self, y, yhat):
        """calcul le gradient du cout par rapport yhat"""
        pass

#########################################################################################################
############################################# CLASSE MODULE #############################################
#########################################################################################################
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

#########################################################################################################
############################################# CLASSE MSELOSS #############################################
#########################################################################################################

class MSELoss(Loss):
    """implementation de la classe Mean square loss"""
    def forward(self, y, yhat):
        return np.linalg.norm( y -yhat, axis=1) ** 2

    def backward(self, y, yhat):
        return -2 * (y-yhat)

#########################################################################################################
############################################# CLASSE LINEAR #############################################
#########################################################################################################

class Linear(Module):
    """Implementation du module Lineaire """
    def __init__(self,input,output):
        self.input = input 
        self.output = output 
        self._parameters = np.random.rand(self.input, self.output) 
        self._gradient = np.zeros((input,output))
        
    def zero_grad(self):
        """Annule gradient"""
        self._gradient = np.zeros(self._gradient.shape) 

    def forward(self, X):
        """Calcule la passe forward, calcul des sorties en fonctions des entrees X"""
        return np.dot(X, self._parameters)

    def update_parameters(self, gradient_step=1e-3):
        """Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step"""
        self._parameters = self._parameters - gradient_step*self._gradient 

    def backward_update_gradient(self, input, delta):
        """Met a jour la valeur du gradient"""
        self._gradient = self._gradient + np.dot( input.T, delta ) 

    def backward_delta(self, input, delta):
        """Calcul la derivee de l'erreur"""
        return np.dot(delta,self._parameters.T) 

#########################################################################################################
############################################ CLASSE SIGMOIDE ############################################
#########################################################################################################

class Sigmoide(Module):
    """ Implementation du module sigmoide"""
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        """Annule gradient"""
        pass

    def forward(self, X):
        """Calcule la passe forward"""
        return (1 / (1 + np.exp(-X)))

    def update_parameters(self, gradient_step=1e-3):
        """Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step"""
        # self._parameters -= gradient_step*self._gradient
        pass

    def backward_update_gradient(self, input, delta):
        """Met a jour la valeur du gradient"""
        pass

    def backward_delta(self, input, delta):
        """Calcul la derivee de l'erreur"""
        return ( np.exp(-input) / ( 1 + np.exp(-input) )**2 ) * delta 

#########################################################################################################
############################################## CLASSE TANH ##############################################
#########################################################################################################

class Tanh(Module):
    """Implementation du module TanH"""
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        """Annule gradient"""
        pass

    def forward(self, X):
        """Calcule la passe forward"""
        return np.tanh(X)

    def update_parameters(self, gradient_step=1e-3):
        """Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step"""
        # self._parameters -= gradient_step*self._gradient
        pass

    def backward_update_gradient(self, input, delta):
        """Met a jour la valeur du gradient"""
        pass

    def backward_delta(self, input, delta):
        """Calcul la derivee de l'erreur"""
        return ( 1 - np.tanh(input)**2 ) * delta

#########################################################################################################
################## ################## ##### FONCTIONS TESTS ################## ################## ######
#########################################################################################################

# TEST PARTIE 1: LINEAIRE MODULEs

# TEST PARTIE 2: NON LINEAIRE MODULE





