#import matplotlib.pyplot as plt
import numpy as np

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

def create_plot():
    data = {'a': np.arange(50), 'c': np.random.randint(0, 50, 50),'d': np.random.randn(50)}
    data['b'] = data['a'] + 10 * np.random.randn(50)
    data['d'] = np.abs(data['d']) * 100
    #plt.scatter('a', 'b', c='c', s='d', data=data)
    #plt.xlabel('entry a')
    #plt.ylabel('entry b')
    #plt.savefig('static/images/test.png')

# takes the graph number and returns the string of the graph name
def image_name_finder(number):
    image_name = 'test'
    return image_name

#create_plot()
