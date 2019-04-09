#import matplotlib.pyplot as plt
import numpy as np
import database
import predictor as p
import graph as g

graph_dict = {1:'Age',2:'Sex',3:'Chest Pain Type',4:'Resting Blood Pressure',\
    5:'Serum Cholestrol',6:'Fasting Blood Sugar',7:'Resting ECG Results',\
        8:'Max. Heart Rate', 9:'Exercise Induced Angina', 10:'oldpeak',\
            11:'Peak Exercise ST slope',12:'Major vessels by flourosopy',\
                13:'thal',14:'target'}

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

def database_startup():
    db = database.server_database()
    db.import_data()
    return db

db = database_startup()

def result_converter(dict_result):
    result = []
    for val in dict_result:
        print(dict_result[val])
        # Aiden, can you convert to the values you expect here
        # then append to result and it should then take care of it
        # data should be already validated
    return result

def predicted(dict_result):
    data = result_converter(dict_result)
    predictor = p.Predictor()
    result = predictor.nnPredict(data)
    return result

def create_graphs(db):
    # Create graphs and save files

    # Chest pain type
    g3 = g.graph(3, db)
    g3.g.create_plot_3()

    # Resting Blood Pressure
    g4 = g.graph(4, db)
    g4.g.create_plot_4()

    # Serum Cholestrol
    g5 = g.graph(5, db)
    g5.g.create_plot_5()

    # Fasting Blood Sugar
    g6 = g.graph(6, db)
    g6.g.create_plot_6()

    # Resting ECG
    g7 = g.graph(7, db)
    g7.g.create_plot_7()

    # Max Heart Rate
    g8 = g.graph(8, db)
    g8.g.create_plot_8()

    # Exercise Induced Angina
    g9 = g.graph(9, db)
    g9.g.create_plot_9()

    # Oldpeak


    # Slope
    g11 = g.graph(11,db)
    g11.g.create_plot_11()

    # Number Vessels Coloured by Flouroscopy
    g12 = g.graph(12,db)
    g12.g.create_plot_12()

    # Thalassemia
    g13 = g.graph(13, db)
    g13.g.g.create_plot_13()