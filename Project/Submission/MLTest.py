'''
    Test File Where I will play with ideas on creating a NN Model, Training it and predicting values etc.
    When A working Idea is found, will transer to a separate file and clean up code
'''

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

## need to read db for cleaned data
def pcaStuff():
    labels = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'serum_cholesterol', 'fasting_blood_sugar', 'resting_electro', 'max_heart_rate', 'exercise_angina', 'oldpeak', 'slope_of_peak', 'no_of_colored_vessels', 'thal', 'target']
    data = pd.read_csv("static/processed.cleveland_no_headers.csv", names=labels)
    data = data.replace('?', pd.np.nan).dropna()
    data['thal'] = np.where(data.thal == '7.0', 'Reversible Defect', data.thal)
    data['thal'] = np.where(data.thal == '6.0', 'Fixed Defect', data.thal)
    data['thal'] = np.where(data.thal == '3.0', 'Normal', data.thal)


    X = data.drop('thal', 1)
    y = data['thal']
    accuracy = []

    for i in range(len(labels)-1):
        print("******PCA Components = {}*******".format(i+1))
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        pca = PCA(n_components=i+1)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        '''print(pca.explained_variance_ratio_)
        i = 0
        ratio = 0
        while ratio < .95 and i < len(pca.explained_variance_ratio_):
            ratio += pca.explained_variance_ratio_[i]
            i += 1
        
        print("I = {} and Variance = {}".format(i,ratio))'''

        classifier = RandomForestClassifier(max_depth=1, random_state=0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        #print(cm)
        #print(accuracy_score(y_test, y_pred))
        #print(y_pred)
        accuracy.append(accuracy_score(y_test, y_pred))
    
    plt.plot(accuracy)
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Accuracy of Predicting Thal Levels")
    #plt.show()

### need to read db for cleaned data
def pcaStuff2():
    labels = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'serum_cholesterol', 'fasting_blood_sugar', 'resting_electro', 'max_heart_rate', 'exercise_angina', 'oldpeak', 'slope_of_peak', 'no_of_colored_vessels', 'thal', 'target']
    data = pd.read_csv("static/processed.cleveland_no_headers.csv", names=labels).drop('target',1)
    data = data.replace('?', pd.np.nan).dropna()
    data['thal'] = np.where(data.thal == '7.0', 'Reversible Defect', data.thal)
    data['thal'] = np.where(data.thal == '6.0', 'Fixed Defect', data.thal)
    data['thal'] = np.where(data.thal == '3.0', 'Normal', data.thal)


    X = data.drop('thal', 1)
    y = data['thal']
    legendlabels = []

    for i in range(len(labels)-2):
        accuracy = []
        xaxis = []
        #print("******PCA Components = {}*******".format(i+1))
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        pca = PCA(n_components=i+1)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        '''print(pca.explained_variance_ratio_)
        i = 0
        ratio = 0
        while ratio < .95 and i < len(pca.explained_variance_ratio_):
            ratio += pca.explained_variance_ratio_[i]
            i += 1
        
        print("I = {} and Variance = {}".format(i,ratio))'''
        for j in range(len(labels)-2):
            print('*********PCA {} & Depth {}**********'.format(i+1,j+1))
            classifier = RandomForestClassifier(max_depth=j+1, random_state=0)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            cm = confusion_matrix(y_test, y_pred)
            #print(cm)
            #print(accuracy_score(y_test, y_pred))
            #print(y_pred)
            accuracy.append(accuracy_score(y_test, y_pred))
            xaxis.append(j+1)
        plt.plot(xaxis,accuracy)
        legendlabels.append("PCA{}".format(i+1))

    
    plt.xlabel("Depth of Tree")
    plt.ylabel("Accuracy of Predicting Thal Levels")
    plt.legend(legendlabels,loc='upper center',ncol=4,bbox_to_anchor=[0.5, 1.15])
    #plt.show()

## need to read cleaned data db
def allPlots():
    labels = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'serum_cholesterol', 'fasting_blood_sugar', 'resting_electro', 'max_heart_rate', 'exercise_angina', 'oldpeak', 'slope_of_peak', 'no_of_colored_vessels', 'thal', 'target']
    data = pd.read_csv("static/processed.cleveland_no_headers.csv", names=labels)
    data = data.replace('?', pd.np.nan).dropna()
    '''data['thal'] = np.where(data.thal == '7.0', 'Reversible Defect', data.thal)
    data['thal'] = np.where(data.thal == '6.0', 'Fixed Defect', data.thal)
    data['thal'] = np.where(data.thal == '3.0', 'Normal', data.thal)
    '''

    hdColorMap = {1:'red', 0:'green'}
    hdLabelMap = {1:'Heart Disease', 0:'No Heart Disease'}
    thalColorMap = {3.0:'green', 6.0:'yellow',7.0:'red'}
    thalLabelMap = {3:'Normal', 6: 'Fixed Defect', 7: 'Reversible Defect'}
    fig = plt.figure()
    scaler = StandardScaler()


    sharedPlots = []

    #Add Shared Subplots
    for row in range(len(labels[:9])):
        sharedPlots.append([])
        for col in range(len(labels[:9])):
            sharedPlots[row].append(fig.add_subplot(len(labels[:9]),len(labels[:9]), (row*len(labels[:9])+col+1)))
            x = np.array(data[labels[col]]).reshape(-1,1)
            x = scaler.fit_transform(x)
            y = np.array(data[labels[row]]).reshape(-1,1)
            y = scaler.fit_transform(y)
            thal = data['thal'].astype('float64')
            plot = sharedPlots[row][col]
            for t in np.unique(thal):
                ix = np.where(thal == t)
                plot.scatter(x[ix], y[ix], c = thalColorMap[t], label = thalLabelMap[t], s = 1)

    plt.xlabel("\t".join(labels[:9]),horizontalalignment='center')
    #plt.show()

def pytorchTest():
    x = torch.ones(2,2, requires_grad=True)
    print(x)

    y = x + 2
    print(y)

    z = y*y*3
    out = z.mean()
    print(z,out)

    a = torch.randn(2,2)
    a = ((a*3)/(a-1))
    #print(a.requires_grad)
    a.requires_grad_(True)
    #print(a.requires_grad)
    b = (a*a).sum()
    #print(b.grad_fn)
    #print(b)

    out.backward()
    print(x.grad)

### need to read cleaned data db
def neuralNetworkLearning():
    labels = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'serum_cholesterol', 'fasting_blood_sugar', 'resting_electro', 'max_heart_rate', 'exercise_angina', 'oldpeak', 'slope_of_peak', 'no_of_colored_vessels', 'thal', 'target']
    data = pd.read_csv("static/processed.cleveland_no_headers.csv", names=labels)
    data = data.replace('?', pd.np.nan).dropna()
    data['target'] = np.where(data.target == 0, 0, 1)

    yArr = np.array(data['target']).astype('int')
   
    scaler = MinMaxScaler()

    numpyYTest = yArr[:50]
    numpyYTrain = yArr[50:]
    y = torch.tensor(numpyYTrain, dtype=torch.long)
    yTest = torch.tensor(numpyYTest, dtype=torch.long)
    data = data.drop(['target'], axis=1)

    numpy_data = scaler.fit_transform(np.array(data).astype('float'))
    print(numpy_data)

    X = torch.Tensor(numpy_data[50:])
    XTest = torch.Tensor(numpy_data[:50])

    #N is number of samples, F_in is number of features
    #H is hidden dimensions, F_out is output features (1 the heart diesease),
    N, F_in, H1, H2, H3,  F_out = len(y), 13, 7, 7, 7, 2
    dataModel = nn.Sequential(
        nn.Linear(F_in, H1),
        nn.Linear(H1,H2),
        nn.Linear(H2,H3),
        nn.Sigmoid(),
        nn.Linear(H3, F_out),
        nn.LogSoftmax()
    )
    
    #learningRate = 0.01
    for s in range(1):

        optimizer = optim.Adam(dataModel.parameters())
        criterion = nn.NLLLoss()
        accuracyList = []
        for t in range(1000):
            optimizer.zero_grad()
            y_pred = dataModel(X)

            loss = criterion(y_pred, y)
            print("{}: {} {}".format(s,t, loss.item()))

            loss.backward()
            optimizer.step()

            '''with torch.no_grad():
                for param in dataModel.parameters():
                    param -= learningRate * param.grad'''

            y_pred = dataModel(XTest)
            predicted = y_pred.detach().numpy()
            predicted = [np.argmax(x) for x in predicted]
            loss = criterion(y_pred, yTest)
            matches = np.where(predicted==numpyYTest,1,0)
            correctness = np.count_nonzero(matches)/len(matches)*100
            accuracyList.append(correctness)
            if(correctness >= 94):
                break

        print("Predicted Target: " , predicted)
        print("Actual Target: " , numpyYTest)
        print("Array of Matches: " , matches)
        np.set_printoptions(formatter={'float_kind':'{:f}'.format})
        probabilities = np.exp(y_pred.detach().numpy())
        print(probabilities)
        print("Model Accuracy Test: {}%".format(correctness))
        y_pred = dataModel(XTest)
        predicted = y_pred.detach().numpy()
        predicted = [np.argmax(x) for x in predicted]
        loss = criterion(y_pred, yTest)
        matches = np.where(predicted==numpyYTest,1,0)
        correctness = np.count_nonzero(matches)/len(matches)*100
        print("Model Accuracy Train: {}%".format(correctness))

        print("{}: Most Accurate at t={}".format(s,np.argmax(accuracyList)))
        plt.plot(accuracyList)

    plt.xlabel("Training Count")
    plt.ylabel("Model Accuracy (%)")
    #plt.show()
    
#allPlots()
neuralNetworkLearning()