from linear import Linear
from MSELoss import MSELoss
import TanH,Sigmoide,Sequentiel,projet_etu,Optim
from sklearn.datasets import make_blobs,make_moons
from matplotlib import pyplot as plt
from pandas import DataFrame
import numpy as np 

# generate 2d classification dataset
X, y = make_moons(n_samples=100, noise=0.1)
# scatter plot, dots colored by class value
# df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
# colors = {0:'red', 1:'blue'}
# fig, ax = pyplot.subplots()
# grouped = df.groupby('label')
# for key, group in grouped:
#     group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
# pyplot.show()

# TEST PARTIE 1 
# module lineaire et mseloss
# module_lineaire = Linear(X.shape[1],4)
# print(module_lineaire.zero_grad())
# print(module_lineaire.forward(X))


def neural_network(X_train, y_train, nombre_neurone, learning_rate=0.01, n_iter=1000, model = Linear, f_cout = MSELoss):
    modele = model(X_train.shape[1],nombre_neurone)
    cout = f_cout()
    # y_train = y_train.reshape((-1,1))
    
    train_loss = []
    train_accuracy = []

    for i in range(n_iter):
        # print(i)
        y_hat = modele.forward(X_train)
        last_delta = cout.backward(y_train,y_hat)
        delta = modele.backward_delta(X_train,last_delta)
        # print(X_train.shape,delta.shape,modele._gradient.shape)
        modele.backward_update_gradient(X_train,delta)
        modele.update_parameters(learning_rate)
        modele.zero_grad()
        train_loss.append(cout.forward(y_train,y_hat).mean())
    
    return modele, train_loss

modele, train_loss = neural_network(X,y,2)
print(modele._gradient)
print(np.array(train_loss).shape)
print(len(train_loss))
# print(train_loss)
plt.figure()
plt.plot(range(len(train_loss)),train_loss)
plt.show()




