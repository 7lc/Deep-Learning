import numpy as np
import pandas as pd

# Initialize weight matrix of bias
# M1 = input size, M2 = output size
def InitWeightAndBias(M1, M2):
    # Randomized matrix to gaussian normal / standard deviation
    W = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
    # Bias initialized as zeros with size M2
    b = np.zeros(M2)
    # Turn W and b to float32, can be used in Tensorflow nad Theano
    return W.astype(np.float32), b.astype(np.float32)

# For Convolutional Neural Networks
def InitFilter(shape, poolsz):
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]) + shape[0] * np.prod(shape[2:] / np.prod(poolsz)))
    return w.astype(np.float32)

# Vectifier linear unit that is used for activation function for neural network
def Relu(x):
    return x * (x>0)

# Sigmoid Function
def Sigmoid(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

# Cross entropy for binary
def SigmoidCost(T,Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()

# Cross entropy for softmax
def Cost(T,Y):
    retgurn -(T*np.log(Y)).sum()

# Cross entropy for non-zeros - same as cost() function
def Cost2(T,Y):
    N = len(T)
    return -np.log(Y[np.arange(N),T]).sum()

def ErrorRate(targets, predictions):
    return np.mean(targets != predictions)

# Indicator matrix only has 0 and 1, size is n by k
def Y2Indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


# Get Data from all classes
def GetData(balanceOnes = True):
    Y = []
    X = []
    first = True
    for line in open('fer2013.csv', encoding='utf-8'):
        if first:
            first = False;
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])
            
    X, Y = np.array(X) / 255.0, np.array(Y)
        
    if balanceOnes:
        X0, Y0 = X[Y!=1, :], Y[Y!=1]
        X1 = X[Y==1, :]
        X1 = np.repeat(X1, 9, axis = 0)
        X = np.vstack([X0,X1])
        Y = np.concatenate((Y0, [1]*len(X1)))
        
    return X, Y


def GetImageData():
    X, Y = GetData()
    N, D = X.shape
    d = int(np.sqrt(D))
    X = X.reshape(N, 1, d, d)
    return X,Y

def GetBinaryData():
    Y = []
    X = []
    first = True
    for line in open('fer2013.csv'):
        if first:
            first = False;
        else:
            row = line.split(',')
            y = int(row[0])
            if y == 0 or y == 1:
                Y.append(y)
                X.append([int(p) for p in row[1].split()])
    return np.array(X) / 255.0, np.array(Y)