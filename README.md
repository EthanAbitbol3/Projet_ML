# Machine Learning Project : Neural Network Library

**Abstract :** In this project, I have to develop a neural network library in Python. The implementation is based on the older version of Pytorch, before the autograd. All the sections of the neural network designed are divided into modules. These one could be **linear** or **activation function**. I have also develop **cost functions**. Here is a representation of the neural network.


<img src="https://github.com/jdufou1/M1ANDROIDE_ML/blob/main/projet/img/img1.png" alt="drawing" width="100%"/>

## Modules

**Linear module :** you can find the linear module here `./lib/module/Linear.py`. The linear allows to make the forward and backward transition. In this way, you need to compute the gradient and update the parameters in each loop turn of learning. 
\
**Sequential module :** you can find the sequential module here `./lib/module/Sequential.py`. This module were implemented to make the learning more easier. This module puts all the layers of the neural network inside a list. So, with the forward and backward methods, we realize the method for all of the modules.
\
**Conv module :** conv 1D and 2D have been implemented
\
**MaxPool module :** maxpool 1D and 2D have been implemented
\
**Flatten module :** This layer allows to link the convolutionnal layers with the fully connected layers.

## Optimizer

**Optim :** you can find the optim class here `./lib/module/Optim.py`. The goal of this class is to make all of the learning step in one function. So, you just need to make a loop with the `step` method. Then, you can create other child class to fill the update method.
\
**SGD optimizer :** you can find the SGD optimizer here `./lib/module/Optim.py`. This class implements the stochatic gradient descent.

## Activation functions

Here are the activations functions :
1. Sigmoide : `./lib/module/Sigmoide.py`
2. TanH : `./lib/module/TanH.py`
3. Softmax : `./lib/module/Softmax.py`
4. ReLU : `./lib/module/ReLU`


## Cost functions

Here are the cost functions :
1. MSEloss : `./lib/loss/MSEloss.py`
2. CEloss : `./lib/loss/CEloss.py`
3. CELogSM : `./lib/loss/CELogSM`

## Some experiences


In this part, I show you some experiences with the neural network library that I have developed. These experiences are available in the `./tests` repo.
\
**Regression with simple data :** Here is the data that I have used to train my model.

<div style="display:flex;">
    <img src="https://github.com/jdufou1/M1ANDROIDE_ML/blob/main/projet/img/img_linear_3.png" alt="drawing" width="45%"/>
    <img src="https://github.com/jdufou1/M1ANDROIDE_ML/blob/main/projet/img/img_linear_2.png" alt="drawing" width="45%"/>
</div>

\
How you can see, the decision border fits perfectly with the data thanks to the linear module and the Mean Square Error loss.


**Binary classification with 2 gaussians :** Here is the data that I have used to train my model.

<div style="display:flex;">
    <img src="https://github.com/jdufou1/M1ANDROIDE_ML/blob/main/projet/img/img_activation_2.png" alt="drawing" width="45%"/>
    <img src="https://github.com/jdufou1/M1ANDROIDE_ML/blob/main/projet/img/img_activation_1.png" alt="drawing" width="45%"/>
</div>

\
We see that the decision surface separates perfectly the two gaussians with an accuracy of **100%**.

**Binary classification with the XOR problem :** Here is the data that I have used to train my model.

<div style="display:flex;">
    <img src="https://github.com/jdufou1/M1ANDROIDE_ML/blob/main/projet/img/img_activation_4.png" alt="drawing" width="45%"/>
    <img src="https://github.com/jdufou1/M1ANDROIDE_ML/blob/main/projet/img/img_activation_3.png" alt="drawing" width="45%"/>
</div>

\
The model has an accuracy of **99.4%** with the XOR problem.
\
\
**Multiclass problem with USPS data :** now, we use the following neural network for multiclass problem : 
*Linear(X.shape , 256)-TanH()-Linear(256,128)-TanH()-Linear(128,y.shape)-Softmax()-CEloss* 
\
Here are some pics of digits from USPS Database (16x16)

<div style="display:flex;">
    <img src="https://github.com/jdufou1/M1ANDROIDE_ML/blob/main/projet/img/img_usps_1.png" alt="drawing" width="30%"/>
    <img src="https://github.com/jdufou1/M1ANDROIDE_ML/blob/main/projet/img/img_usps_2.png" alt="drawing" width="30%"/>
    <img src="https://github.com/jdufou1/M1ANDROIDE_ML/blob/main/projet/img/img_usps_3.png" alt="drawing" width="30%"/>
</div>

Now let's see the results of the training :
<div style="display:flex;">
    <img src="https://github.com/jdufou1/M1ANDROIDE_ML/blob/main/projet/img/img_multiclasse_1.png" alt="drawing" width="30%"/>
    <img src="https://github.com/jdufou1/M1ANDROIDE_ML/blob/main/projet/img/img_multiclasse_2.png" alt="drawing" width="30%"/>
    <img src="https://github.com/jdufou1/M1ANDROIDE_ML/blob/main/projet/img/img_multiclasse_3.png" alt="drawing" width="30%"/>
</div>

With this model of neural network , our results are good enough with an accuracy of **91.33%** for the train set and **80.57** for the test set.

## How to use

Firstly , you need to clone the repo \
`git clone https://github.com/jdufou1/M1ANDROIDE_ML.git` \
Then, move in the project repo \
`cd projet`
So, you could find two repos : `./lib` which contains the main class for the neural network construction and `./tests` which contains all the tests for the class file implemented. \
You can also find data about digits in the `./data` repo.
