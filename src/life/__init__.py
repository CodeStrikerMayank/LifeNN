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
Y = data["diagnosis"].map({"malignant":0,"benign":1}).to_numpy()
x_train,x_temp,y_train,y_temp=ts(X,Y,test_size=0.30,random_state=42,shuffle=True)
x_valid,x_test,y_valid,y_test = ts(x_temp,y_temp,test_size=0.50,random_state=42,shuffle=True)
print(x_train.shape)