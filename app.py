from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
import torch
import torchvision # torch package for vision related things
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
#import torchvision.datasets as datasets  # Standard datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
#from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!

from sklearn.preprocessing import StandardScaler

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__() #Override the init method
        # Our first linear layer take input_size, in this case 784 nodes to 50
        # and our second linear layer takes 50 to the num_classes we have, in
        # this case 10.
        self.fc1 = nn.Linear(input_size,11 )
        #self.fc2 = nn.Linear(10, 6)
        self.fc2 = nn.Linear(11, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = (self.fc2(x))
        #x = (self.fc3(x))
        return x

model = torch.load('iris_ml_model')
app = Flask(__name__)

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    Fuel_Type_Diesel=0
    if request.method == 'POST':
        #Year = int(request.form['Year'])
        sepall = int(request.form['sepall'])
        sepalw = int(request.form['sepalw'])
        petall = int(request.form['petall'])
        petalw = int(request.form['petalw'])
        data =torch.FloatTensor ([sepall,sepalw,petall,petalw])
        scores = model(data)
            #print(scores)
        predictions = torch.argmax(scores)
 #classific = {'Setosa': 0,'Versicolor': 1,'Virginica':2}
        if (predictions==0):
            flower = 'Setosa'
        if (predictions==1):
            flower = 'Versicolor'
        if (predictions==2):
            flower = 'Virginica'
                
        return render_template('index.html',prediction_text="The flower is {}".format(flower))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
