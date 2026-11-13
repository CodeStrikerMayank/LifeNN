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
def sigmoid(z):
    return 1/(1+(np.exp(-np.clip(z,-500,500))))
def relu(z):
    return np.maximum(z*0.01,z)
def relu_deri(z):
    return np.where(z>0,1.0,0.01)

def process(x_train,y_train,x_valid,y_valid,x_test,y_test):
    # Creating neurons and their layers
    a1 = np.random.randn(30,16)*np.sqrt(2/4)
    b1 = np.zeros((1,16))
    a2 = np.random.randn(16,8)*np.sqrt(2/4)
    b2 = np.zeros((1,8))
    a3 = np.random.randn(8,4)*np.sqrt(2/4)
    b3 = np.zeros((1,4))
    a4 = np.random.randn(4,2)*np.sqrt(2/4)
    b4 =np.zeros((1,2))
    a5 = np.random.randn(2,1)*np.sqrt(2/4)
    b5= np.zeros((1,1))
    
    