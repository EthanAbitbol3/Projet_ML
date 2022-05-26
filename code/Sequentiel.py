"""
Classe Sequentiel
ABITBOL YOSSEF
DUFOURMANTELLE JEREMY
"""
from projet_etu import Module 

from collections import OrderedDict

import numpy as np

class Sequentiel(Module):

    def __init__(self, *args):
        """Initialisation des parametres"""
        self._parameters = None
        self._gradient = None
        self._modules = OrderedDict()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for cle, module in args[0].items():
                 self._modules[cle] = module
        else:
            for index, module in enumerate(args):
                self._modules[str(index)] = module

    def zero_grad(self):
        """Annule gradient de tous les modules"""
        for module in self._modules.values():
            module.zero_grad()

    def forward(self, X):
        """Calcule la passe forward de tous les modules"""
        input=X
        for module in self._modules.values():
            input = module.forward(input)
        self._forward=input
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
        """Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step"""
        for module in self._modules.values():
            module.update_parameters(gradient_step)

    def backward_update_gradient(self, input, delta):
        """Met a jour la valeur du gradient pour tous les modules"""
        modules=list(self._modules.values())[::-1]
        suiv = modules[0]
        for module in modules[1:]:
            prec=module
            suiv.backward_update_gradient(prec._forward,delta)
            delta=suiv._delta
            suiv=prec
        suiv.backward_update_gradient(input, delta)

    def backward_delta(self, input, delta):
        """Calcul la derivee de l'erreur"""
        modules = list(self._modules.values())[::-1]
        suiv = modules[0]
        for module in modules[1:]:
            prec = module
            delta = suiv.backward_delta(prec._forward, delta)
            suiv = prec
        self._delta = suiv.backward_delta(input,delta)
        return self._delta


    def predict(self,data) : 
        return np.where(  self.forward(data) >= 0.5 , 1 , 0 )