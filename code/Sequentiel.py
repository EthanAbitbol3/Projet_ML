from sys import modules
import numpy as np 

class Sequentiel():
    """Implementation de la classe Sequentiel"""
    def __init__(self,modules,loss) -> None:
        self.modules = modules # liste des differents modules
        self.loss = loss # fonctions loss
        self.list_error = []

    def fit(self,X, y):
        # if biais == True:
        #     bias = np.ones((len(X), 1))
        #     X = np.hstack((bias, X))

        # partie forward 
        res = []
        res.append(self.modules[0].forward(X))
        for i in range (1,len(self.modules)):
            res.append(self.modules[i].forward(res[-1])) # on utilise le res du forward precedent 

        self.list_error.append(np.sum(self.loss.forward(y, res[-1])))

        # partie backward 
        delta = []
        delta.append(self.loss.backward( y, res[-1] ))
        modules_inverse = self.modules[::-1]
        res_inverse = res[::-1]
        for i in range(0,len(modules_inverse)-1): # on parcours 
            delta.append(modules_inverse[i].backward_delta( res_inverse[i+1], delta[-1] ) )
        
        return res, delta , self.list_error

    def predict(self,xtest,biais = True):
        if biais == True:
            bias = np.ones((len(xtest), 1))
            xtest = np.hstack((bias, xtest))
        
        res1 = self.linear_1.forward(xtest)
        res2 = self.tanh.forward(res1)
        res3 = self.linear_2.forward(res2)
        res4 = self.sigmoide.forward(res3)

        return np.argmax(res4, axis = 1) 