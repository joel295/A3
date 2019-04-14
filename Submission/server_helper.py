#import matplotlib.pyplot as plt
import numpy as np
import database, sqlite3
import predictor as p
import graph as g

graph_dict = {1:'Age',2:'Sex',3:'Chest Pain Type',4:'Resting Blood Pressure',\
    5:'Serum Cholesterol',6:'Fasting Blood Sugar',7:'Resting ECG Results',\
        8:'Maximum Heart Rate', 9:'Exercise Induced Angina', 10:'Oldpeak',\
            11:'Peak Exercise ST Slope',12:'Major Vessels by Flouroscopy',\
                13:'Thalassemia',14:'target', 15:'3D - Max Heart Rate vs Serum Cholesterol',\
                    16:'2D - Max Heart Rate vs Serum Cholesterol', 17:'3D - Max Heart Rate vs Oldpeak',\
                        18:'2D - Max Heart Rate vs Oldpeak'}

def validate_result(result):
    ## check the values being used here
    hold = []
    try:
        if int(result['age']) < 0 or int(result['age'])> 120:
            hold.append('Age')
    except ValueError:
        hold.append('Age')
    try:
        if int(result['resting_blood_pressure']) < 10 or int(result['resting_blood_pressure']) > 300:
            hold.append('Resting blood pressure')
    except ValueError:
        hold.append('Resting blood pressure')
    try:
        if int(result['serum_cholestoral']) < 0 or int(result['serum_cholestoral']) > 500:
            hold.append('Serum Cholestoral')
    except ValueError:
        hold.append('Serum Cholestoral')
    try:
        if int(result['max_heart']) < 0 or int(result['max_heart']) > 300:
            hold.append('Maximum Heart Rate Achieved')
    except ValueError:
        hold.append('Maximum Heart Rate Achieved')
    try:
        if float(result['oldpeak']) < 0 or float(result['oldpeak']) > 10:
            hold.append('Oldpeak')
    except ValueError:
        hold.append('Oldpeak')
    if not hold:
        return False
    else:
        return hold

def test_list():
    result = []
    for i in range(1,15):
        result.append([i,graph_dict[i],i*2])
    return result

# takes the graph number and returns the string of the graph name
def image_name_finder(number):
    if (number != None):
        image_name = 'graph' + str(number)
    else:
        return False
    return image_name

# as above, but for graphs by heart disease
def labelled_image_name_finder(number = None):
    print(number)
    if (number != None):
        image_name = 'graph' + str(number) + '_labelled'
    else:
        image_name = 'test'
    return image_name


def database_startup():
    db = database.server_database()
    db.import_data()
    return db

db = database_startup()

def result_converter(dict_result):
    result = []
    allLabels = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'serum_cholestoral', 'fasting_bs', 'resting_ecg','max_heart', 'ei_angina', 'oldpeak', 'peak_est', 'colours', 'thal']

    for label in allLabels:
        result.append(float(dict_result[label]))
    return result

def predicted(dict_result):
    data = result_converter(dict_result)
    predictor = p.Predictor()
    result = predictor.nnPredict(data)
    return result