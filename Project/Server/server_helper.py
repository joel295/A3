import matplotlib.pyplot as plt
import numpy as np

graph_dict = {1:'Age',2:'Sex',3:'Chest Pain Type',4:'Resting Blood Pressure',\
    5:'Serum Cholestrol',6:'Fasting Blood Sugar',7:'Resting ECG Results',\
        8:'Max. Heart Rate', 9:'Exercise Induced Angina', 10:'oldpeak',\
            11:'Peak Exercise ST slope',12:'Major vessels by flourosopy',\
                13:'thal',14:'target'}


def test_list():
    result = []
    for i in range(15):
        result.append([i*1,i*2,i*3])
    return result

def create_plot():
    data = {'a': np.arange(50), 'c': np.random.randint(0, 50, 50),'d': np.random.randn(50)}
    data['b'] = data['a'] + 10 * np.random.randn(50)
    data['d'] = np.abs(data['d']) * 100
    plt.scatter('a', 'b', c='c', s='d', data=data)
    plt.xlabel('entry a')
    plt.ylabel('entry b')
    plt.savefig('static/images/test.png')

create_plot()

'''
<li class="active"></li>
<li><a href="/">Home</a></li>
<li><a href="/graph?=1">Age</a></li>
<li><a href="/graph?=2">Sex</a></li>
<li><a href="/graph?=3">Chest Pain Type</a></li>
<li><a href="/graph?=4">Resting Blood Pressure</a></li>
<li><a href="/graph?=5">Serum Cholestrol</a></li>
<li><a href="/graph?=6">Fasting Blood Sugar</a></li>
<li><a href="/graph?=7">Resting ECG Results</a></li>
<li><a href="/graph?=8">Max. Heart Rate</a></li>
<li><a href="/graph?=9">Exercise Induced Angina</a></li>
<li><a href="/graph?=10">oldpeak</a></li>
<li><a href="/graph?=11">Peak Exercise ST slope</a></li>
<li><a href="/graph?=12">Major vessels by flourosopy</a></li>
<li><a href="/graph?=13">thal</a></li>
<li><a href="/graph?=14">target</a></li>
'''