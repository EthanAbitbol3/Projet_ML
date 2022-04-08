from sys import modules
import numpy as np 

class Sequentiel():
    """Implementation de la classe Sequentiel"""
    def __init__(self,modules,loss) -> None:
        self.modules = modules # liste des differents modules
        self.loss = loss # fonctions loss
        self.list_error = None
        self.res = []
        self.delta = []

    def fit(self,X, y):

        # partie forward 
        self.res.append(self.modules[0].forward(X))
        for i in range (1,len(self.modules)):
            self.res.append(self.modules[i].forward(self.res[-1])) # on utilise le res du forward precedent 

        self.error = np.sum(self.loss.forward(y, self.res[-1]))

        # partie backward 
        self.delta.append(self.loss.backward( y, self.res[-1] ))
        modules_inverse = self.modules[::-1]
        res_inverse = self.res[::-1]
        for i in range(0,len(modules_inverse)-1): # on parcours 
            self.delta.append(modules_inverse[i].backward_delta( res_inverse[i+1], self.delta[-1] ) )
        

    def predict(self,xtest,biais = True):
        if biais == True:
            bias = np.ones((len(xtest), 1))
            xtest = np.hstack((bias, xtest))

        res_pred = []
        res_pred.append(self.modules[0].forward(xtest))
        for i in range (1,len(self.modules)):
            res_pred.append(self.modules[i].forward(res_pred[-1])) # on utilise le res du forward precedent 
        
        return np.argmax(res_pred[-1], axis = 1) 