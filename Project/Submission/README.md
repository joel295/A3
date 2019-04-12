# COMP9321 Assignment 3

### Group Project - Heart Disease

This project was designed and built by
- Aiden Meikle - z5229381
- Joel Lawrence - z3331029
- Kaylen Payer - z5076219
- Thomas Welsh - z5016234

#### Overview:

This project is a web application with the following functions:
- Visualize statistics of heart disease data by age and sex
- Show the potential important factors related to heart disease
- Predict heart disease (and the accuracy of the prediction) based on user input values for various health attributes

#### Dataset:

This project uses data from the University of California Irvine which has been cleansed.  
This data contains the following attributes:
```
1. Age
2. Sex (1 = male; 0 = female)
3. Chest Pain Type (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 4 = asymptomatic)
4. Resting Blood Pressure
5. Serum Cholesterol (mg/dl)
6. Fasting Blood Sugar > 120 mg/dl (0 = False; 1 = True)
7. Resting Electrocardiographic Results (0 = normal; 1 = ST-T wave abnormality; 2 = probable ventricular hypertrophy)
8. Maximum Heart Rate (bpm)
9. Exercise Induced Angina (0 = False; 1 = True)
10. Oldpeak (ST depression induced by exercice relative to rest) (mV)
11. Slope of the peak exercise ST segment
12. Number of major vessels (0-3) coloured by flouroscopy
13. Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
14. Target (Heart Disease) (0 = False; 1 = True)
```

Source: http://archive.ics.uci.edu/ml/datasets/heart+disease

#### Requirements:

This project uses Python3.7 and requires the following libraries and their dependencies:

- Flask 1.0.2
- Jinja2 2.10.1
- matplotlib 3.0.3
- numpy 1.16.2
- pandas 0.24.2
- scikit-learn 0.20.3
- scipy 1.2.1
- seaborn 0.9.0
- torch 1.0.1.post2
- torchvision 0.2.2.post3

All requirements can be installed with the following command:

`pip3 install -r requirements.txt`

#### Usage:

```
1. Activate a virtual environment

2. Install requirements:  
    $ pip3 install -r requirements.txt

3. Run the server from within the Submission directory:   
    $ python3 server.py

4. In your web browser, visit `127.0.0.1:5000`

5. Navigate using the menu:
    - Visualisations - statistics for each attribute by age and sex
    - Factor Ranking - importance of each attribute for heart disease prediction
    - Prediction - accepts user input to predicts heart disease
    - Graphs by Heart Disease - statistics for each attribute by heart disease
```

#### Github Link:

This project can be found at: https://github.com/joel295/A3
