lazy import pandas as pd 
lazy import numpy as np 
lazy from sklearn.model_selection import train_test_split as ts 

data = pd.read_csv(r"D:\MachineLearning\Core Projects\Life\neural_network_practice_dataset.csv")
information = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension']

X = data[information].to_numpy()
Y = data["diagnosis"].map({"malignant":0,"benign":1}).to_numpy().reshape(-1, 1)
x_train,x_temp,y_train,y_temp=ts(X,Y,test_size=0.30,random_state=42,shuffle=True)
x_valid,x_test,y_valid,y_test = ts(x_temp,y_temp,test_size=0.50,random_state=42,shuffle=True)
"""Standardisation of the  inputs and create a uniformity"""
sig = np.std(x_train,axis=0,keepdims=True)
avg = np.mean(x_train,axis=0,keepdims=True)
x_train = (x_train-avg)/(sig+(1e-8))
x_valid = (x_valid-avg)/(sig+(1e-8))
x_test = (x_test-avg)/(sig+(1e-8))
def sigmoid(z):
    return 1/(1+(np.exp(-np.clip(z,-500,500))))
def relu(z):
    return np.maximum(z*0.01,z)
def relu_deri(z):
    return np.where(z>0,1.0,0.01)
    
def bse(y_true, y_pred):
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return -np.mean(y_true * np.log(y_pred_clipped) + (1.0 - y_true) * np.log(1.0 - y_pred_clipped))
def process(x_train,y_train,x_valid,y_valid,x_test,y_test):
    # Creating neurons and their layers
    w1 = np.random.randn(30, 16) * np.sqrt(2 / 30)
    b1 = np.zeros((1, 16))
    w2 = np.random.randn(16, 8) * np.sqrt(2 / 16)
    b2 = np.zeros((1, 8))
    w3 = np.random.randn(8, 4) * np.sqrt(2 / 8)
    b3 = np.zeros((1, 4))
    w4 = np.random.randn(4, 2) * np.sqrt(2 / 4)
    b4 = np.zeros((1, 2))
    w5 = np.random.randn(2, 1) * np.sqrt(1 / 2)
    b5 = np.zeros((1, 1))    
    lrt = 0.05 
    ephocs = 10000
    # activation the code 
    m = len(x_train)
    for ephoc in range(ephocs):

        a1 = np.dot(x_train,w1)+b1 
        z1 = relu(a1)
        a2 = np.dot(z1,w2)+b2 
        z2 = relu(a2)
        a3 = np.dot(z2,w3)+b3
        z3 = relu(a3)
        a4 = np.dot(z3,w4)+b4 
        z4 = relu(a4)
        a5 = np.dot(z4,w5)+b5
        z5 = sigmoid(a5)

        loss = bse(y_train,z5)

        da5 = (z5-y_train)
        dw5 = (1/m)*(np.dot(z4.T,da5))
        db5 = (1/m)*np.sum(da5,keepdims=True,axis=0)
        da4 = np.dot(da5,w5.T)*relu_deri(a4)
        dw4 = (1/m)*np.dot(z3.T,da4)
        db4 = (1/m)*np.sum(da4,axis=0,keepdims=True)
        da3 = np.dot(da4,w4.T)*relu_deri(a3)
        dw3 = (1/m)*np.dot(z2.T,da3)
        db3 = (1/m)*np.sum(da3,axis=0,keepdims=True)
        da2 = np.dot(da3,w3.T)*relu_deri(a2)
        dw2 = (1/m)*np.dot(z1.T,da2)
        db2 = (1/m)*np.sum(da2,axis=0,keepdims=True)
        da1 = np.dot(da2,w2.T)*relu_deri(a1)
        dw1 = (1/m)*np.dot(x_train.T,da1)
        db1 = (1/m)*np.sum(da1,axis=0,keepdims=True)

        w5 -= lrt*dw5 
        b5 -= lrt*db5 
        w4 -= lrt*dw4 
        b4 -= lrt*db4 
        w3 -= lrt*dw3 
        b3 -= lrt*db3      
        w2 -= lrt*dw2   
        b2 -= lrt*db2 
        w1 -= lrt*dw1  
        b1 -= lrt*db1 

        if ephoc % 1000 == 0:
            print(f"Epoch {ephoc}: Loss = {loss:.4f}")

process(x_train,y_train,x_valid,y_valid,x_test,y_test)         