'''
Course: COMP9321
Group: The Cherry Pies

Module that defines the template for predicitng whether or not
a person has heart disease or not.
2 methods are provided, one is a Neural Network and the Other is a
Random Forest.
'''

from database import *
import os,sys,math
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


#All Predicition Algorithms Stored inside
class Predictor():
    #Initialise object, load neural network/create if it doesn't exist
    #Setup the random forest
    def __init__(self):
        self.heartData = self.loadData()
        self.createScaler()
        self.initialiseNN()
    
    #Load in the heart data from the database, swap 
    def loadData(self):
        try:
            database = server_database()
            data = database.load_all_rows()
            data = data.drop(['ID'], axis = 1)
        except Exception as e:
            print(e)

        return data


    #Initialise our Neural Network, ready to predict values
    def initialiseNN(self):
        #Check if a pre-trained network file already exists
        networkExists = os.path.isfile('neuralnetwork.pt')

        #If network exists, load file, otherwise train and save a new one
        if(networkExists):
            self.dataModel = torch.load('neuralNetwork.pt')
            self.dataModel.eval()
        else:
            self.trainNN()
    
    #Create a min max scaler to scale all input data to the same scale
    def createScaler(self):
        #Need to scale data to uniform [0,1] range
        self.scaler = MinMaxScaler()

        #Remove label data from total data set
        xData = self.heartData.drop(['Target'], axis=1)
        #Initialize the scaler with
        self.scaler.fit(xData)



    def trainNN(self):
        '''
        Method to Train the Neural Network using the First 50 Rows of data for testing and the remaining for
        training the model.
        Returns: Trained Neural Network Model
        '''

        #Target Values are Labels for Training
        yValues = np.array(self.heartData['Target']).astype('int')


        #Use first 50 labels for testing, rest for training
        yTestValues = yValues[:50]
        yTrainValues = yValues[50:]

        #Convert label data to Tensor Objects, used to form Network by pyTorch
        yTrainTensor = torch.tensor(yTrainValues, dtype=torch.long)
        yTestTensor = torch.tensor(yTestValues, dtype=torch.long)
        
        #Remove label data from total data set
        xData = self.heartData.drop(['Target'], axis=1)
        #Scale the heart_disease data into range [0,1]
        xData = self.scaler.transform(np.array(xData).astype('float'))
        

        #Convert the heart_disease data into Tensor Objects
        #These are the inputs to the Neural Network
        xTrainTensor = torch.Tensor(xData[50:])
        xTestTensor = torch.Tensor(xData[:50])
        
        #N is number of samples, F_in is number of features
        #H1 is number of hidden dimensions, F_out is number of output features (2 since 0 is No Disease, 1 is Has Disease)
        F_in, H1, F_out = 13, 7, 2
        #Defines the Structure of the Neural Network
        self.dataModel = nn.Sequential(
            nn.Linear(F_in, H1),    #One linear layer with F_in inputs and H1 outputs
            nn.ReLU(),              #Rectified Linear Unit Activation Function for first layer
            nn.Linear(H1,F_out),    #Second Linear layer with H1 inputs and F_out outputs
            nn.LogSoftmax(dim=1)         #Log Softmax activation function to give probability for each output
        )

        #Use Stochastic Gradient Descent to optimize weights in Model
        optimizer = optim.SGD(self.dataModel.parameters(), lr=0.1, momentum=0.9)
        #Use the Negative Log Loss for the Loss Function, since LogSoftmax Function is used as activation function
        criterion = nn.NLLLoss()
        #Store All accuracies at each 20 batches (batch size of training is 1, so every 20 elements)
        accuracyList = []

        #Peform a max of 1000 training passes on network
        for epoch in range(1000):

            #Resets the gradients of the optimizer
            optimizer.zero_grad()
            #Perform a batch forward pass with the training data
            yPred = self.dataModel(xTrainTensor)

            #Calculate the error loss of these predicted values with the actual values
            loss = criterion(yPred, yTrainTensor)
            #Backpropagate the weights in the neural network
            loss.backward()
            optimizer.step()
            
            #Calculate the Model Accuracy at each step
            yPred = self.dataModel(xTestTensor)
            yPredList = yPred.detach().numpy()
            yPredList = [np.argmax(x) for x in yPredList]
            loss = criterion(yPred, yTestTensor)
            matches = np.where(yPredList==yTestValues,1,0)
            correctness = np.count_nonzero(matches)/len(matches)*100
            accuracyList.append(correctness)
            #If we reach 92% accuracy stop learning
            if(correctness >= 92):
                break


        plt.plot(accuracyList)

        plt.xlabel("Epoch Count")
        plt.ylabel("Model Accuracy (%)")
        plt.savefig('neuralNetworkAccuracy.png')

        torch.save(self.dataModel, 'neuralNetwork.pt')
        self.dataModel.eval()

    def nnPredict(self, attributeList):
        '''
        Given a list of all the attributes, except for Target Value, predict whether this data shows Heart Disease or Not.
        Returns: Tuple[TargetValue, [No_Disease_Probability, Disease_Probability]]
        TargetValue is 0 for No Disease, 1 for Has Disease
        No_Disease_Probability is liklihood of No Disease, vice versa for Disease_Probability.
        A Bar Chart 'diseaseProbability.png' is created too.
        '''


        #Scale the heart_disease data into range [0,1]
        xData = self.scaler.transform(np.array(attributeList).reshape(1,-1).astype('float'))

        #Create a tensor using the attribute data
        xTestTensor = torch.Tensor(xData)

        #Feed the Attribute Values to the Neural Network for prediction
        predictedTensor = self.dataModel(xTestTensor)
        yPrediction = predictedTensor.detach().numpy()
        #Get the index of which output has highest probability (0 for No Disease, 1 for Has Disease)
        predictedValue = np.argmax(yPrediction)

        #Calculate Exponential of Each Probability, Since Log was taken previously
        diseaseProbabilities = [math.exp(x) for x in yPrediction[0]]

        #Colours for each plotted bar, green for highest chance and red for lower chance
        if(diseaseProbabilities[0] > diseaseProbabilities[1]):
            colors = ['green','red']
        else:
            colors = ['red', 'green']
        
        #Create a new figure
        plt.figure()

        #Plot all the data as a bar chart
        plt.bar(['No Heart Disease', 'Heart Disease'], diseaseProbabilities, color = colors)
        plt.title('Heart Disease Prediciton')
        plt.xlabel('Outcome Types')
        plt.ylabel('Probability of Correctnes')
        #Save the figure as a file to be displayed by webpage
        plt.savefig('diseaseProbability.png')

        
        return (predictedValue, diseaseProbabilities)



'''
For Testing Only
To Predict an Outcome first create a Predictor object (1)
Then call obj.nnPredict(List) (2) to predict the outcome of Having Heart Disease
Pass in a List of 13 Attribute Values to nnPredict()

Example:
attrList = [67.0,1.0,4.0,125.0,254.0,1.0,0.0,163.0,0.0,0.2,2.0,2.0,7.0]
predictor = Predictor()
predictedValue, probabilityList = predictor.nnPredict(attrList)
'''