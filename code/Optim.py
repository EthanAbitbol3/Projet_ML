import numpy as np

class Optim():
    """implementation de la classe Optim"""
    def __init__(self,net,loss,eps) -> None:
        self.net = net
        self.loss = loss
        self.eps = eps 