#import matplotlib.pyplot as plt
import numpy as np

graph_dict = {1:'Age',2:'Sex',3:'Chest Pain Type',4:'Resting Blood Pressure',\
    5:'Serum Cholestrol',6:'Fasting Blood Sugar',7:'Resting ECG Results',\
        8:'Max. Heart Rate', 9:'Exercise Induced Angina', 10:'oldpeak',\
            11:'Peak Exercise ST slope',12:'Major vessels by flourosopy',\
                13:'thal',14:'target'}
def validate_function(result):
    ## check the values being used here
    hold = []
    if result['age'] <0 or result['age']> 120:
        hold.append('Age')
    if result['serum_cholestoral'] < 0 or result['serum_cholestoral'] > 1000:
        hold.append('Serum Cholestoral')
    if result['fasting_blood_sugar'] < 120:
        hold.append('Fasting Blood Sugar')
    if result['max_heart'] < 0 or result['max_heart'] > 300:
        hold.append('Maximum Heart Rate Achieved')
    if result['ei_angina'] < 0:
        hold.append('Exercise Induced Angina')
    if result['oldpeak'] < 0:
        hold.append('Oldpeak')
    if result['peak_est'] < 0:
        hold.append('Peak Exercise ST Segment')
    if not hold:
        return False
    else:
        return hold

def test_list():
    result = []
    for i in range(15):
        result.append([i*1,i*2,i*3])
    return result

def create_plot():
    data = {'a': np.arange(50), 'c': np.random.randint(0, 50, 50),'d': np.random.randn(50)}
    data['b'] = data['a'] + 10 * np.random.randn(50)
    data['d'] = np.abs(data['d']) * 100
    #plt.scatter('a', 'b', c='c', s='d', data=data)
    #plt.xlabel('entry a')
    #plt.ylabel('entry b')
    #plt.savefig('static/images/test.png')

#create_plot()
