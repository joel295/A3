## Visualization Plan

#### Project Requirement:

You need to visualize the statistics for basic information (attributes) 3-13 by groups of age and sex with appropriate charts, and display on your web app.

**1. Age**  
    Range: 29-77

**2. Sex**  
    Values: 1 = male, 0 = female

**3. Chest Pain Type**  
    Values: 1 = typical angin, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic

    Series of histograms:
    - Overarching x-axis: chest pain type
    - Overarching y-axis: male, female
    - Histogram x-axis: age range
    - Histogram y-axis: frequency/count


                Type-1  |   Type-2   |  Type-3  |  Type-4  |
              ----------------------------------------------
        count |  graph1  |   graph2  |  graph3  |  graph4  |    male
                  age         age         age        age
              ----------------------------------------------
        count |  graph1  |  graph2   |  graph3  |  graph4  |    female
                  age         age         age        age


**4. Resting Blood Pressure**

    Scatter plot:
    - Different coloured data points for male and female
    - Different colour line of best fit/linear regression for male and female
    - x-axis: age
    - y-axis: blood pressure

**5. Serum Cholesterol**

    Scatter plot: [as above, but y-axis: serum cholesterol]

**6. Fasting Blood Sugar > 120 mg/dl**  
    Values: 0 = false, 1 = true

    Stacked histogram:
    - x-axis: age range
    - y-axis: frequency/count of value = true
    - Stacked bars: male/female


**7. Resting Electrocardiographic Results**  
    Values: 0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy

    Series of histograms: [as attribute 1, but only 3 types]

**8. Max Heart Rate**

    Scatter plot: [as attributes 4,5, but y-axis: heart rate]

**9. Exercise Induced Angina**  
    Values: 0 = false, 1 = true

    Stacked Histogram: [as attribute 6]


**10. Oldpeak**

    Scatter plot: [y-axis: oldpeak]

**11. Slope**

**12. Number Vessels Coloured by Flouroscopy**  
    Values: 1, 2, 3



**13. Thalassemia**  
    Values: 3 = normal, 6 = fixed defect, 7 = reversible defect

    Series of histograms: [as attribute 1]
