'''
Visualization for Heart data attributes 3-13
Creates graph<number>.png files
'''
import os, sys, inspect, sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CURR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURR_DIR)
sys.path.insert(0, PARENT_DIR)

IMG_PATH = PARENT_DIR + '/Submission/static/images/'

class graph:

    def __init__(self, attr_num, db_conn):
        self.name = 'Graph_' + str(attr_num)
        self.db = db_conn


    # Chest Pain Type
    def create_plot_3(self,db):

        # Get data
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex, Chest_Pain_Type FROM Heart;', conn)

        ## Plot histograms male and female by type

        sns.set_style("darkgrid")               # set the background of the charts
        bins = [20,30,40,50,60,70,80]           # group age by 10's
        palette = {'color': ['b', 'r']}         # separate colours for male/female

        g = sns.FacetGrid(df, row='Sex', col='Chest_Pain_Type', hue='Sex',
                margin_titles=True, hue_kws=palette)
        g.map(sns.distplot, 'Age', bins=bins, kde=False,
                hist_kws=dict(edgecolor="black", linewidth=1));

        ## Format Plot

        # Axis Limits
        g.axes[1,0].set_ylim(0,50)
        g.axes[1,0].set_xlim(20,80)
        g.axes[1,0].set_xticks(range(20, 80, 10))
        g.axes[1,0].set_yticks(range(0, 55, 10))

        # Titles
        plt.subplots_adjust(top=0.85, left=0.07, bottom=0.15)
        g.fig.suptitle("Chest Pain Type\n")
        g.axes[0,0].set_title("Typical Angina")
        g.axes[0,1].set_title("Atypical Angina")
        g.axes[0,2].set_title("Non-Anginal Pain")
        g.axes[0,3].set_title("Asymptomatic")

        # Overall Labels
        g.set_axis_labels('', '')
        g.fig.text(x=0.01, y=0.5, verticalalignment='center',s='Frequency', rotation=90, size=11)
        g.fig.text(x=0.5, y=0.01, horizontalalignment='center', s='Age', size=11)
        gender_labels =  ["","","","Male","","","","Female","","","",]
        for i, ax in enumerate(g.axes.flat):
            plt.setp(ax.texts, text=gender_labels[i], size=11)

        g.set_xticklabels(rotation=45, labels=["20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "> 80"])

        # Save image
        plt.savefig(IMG_PATH + 'graph3.png')
        

    # Resting Blood Pressure
    def create_plot_4(self,db):

        # Get data
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex, RBP FROM Heart;', conn)

        # Divide into male and female
        male_df = df.query('Sex == 1')
        female_df = df.query('Sex == 0')

        # Scatter plots
        fig, axes = plt.subplots(nrows=1, ncols=2)
        ax1 = sns.regplot(x=male_df['Age'], y=male_df['RBP'], label='male', ax=axes[0])
        ax2 = sns.regplot(x=female_df['Age'], y=female_df['RBP'], label='female',
                color='red', ax=axes[1])

        ## Format Plot

        # Add titles
        fig.suptitle('Resting Blood Pressure\n')
        ax1.set_title('Male')
        ax2.set_title('Female')
        fig.text(x=0.54, y=0.04, horizontalalignment='center', s='Age', size=11)

        # X-Axis
        ax1.set_xlim(25,84)
        ax2.set_xlim(25,84)
        ax1.set_xticks(range(35, 84, 10))
        ax2.set_xticks(range(35, 84, 10))
        ax1.set_xlabel('')
        ax2.set_xlabel('')

        # Y-Axis
        ax1.set_ylim(90,210)
        ax2.set_ylim(90,210)
        ax1.set_ylabel('RBP')
        ax2.set_ylabel('')
        ax2.set_yticklabels('')

        # Layout
        fig.tight_layout()
        fig.subplots_adjust(top=0.85, bottom=0.15)

        # Save image
        plt.savefig(IMG_PATH + 'graph4.png')


    # Serum Cholesterol
    def create_plot_5(self,db):

        # Get data
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex, Serum_Chol FROM Heart;', conn)

        # Divide into male and female
        male_df = df.query('Sex == 1')
        female_df = df.query('Sex == 0')

        # Scatter plots
        fig, axes = plt.subplots(nrows=1, ncols=2)
        ax1 = sns.regplot(x=male_df['Age'], y=male_df['Serum_Chol'], label='male', ax=axes[0])
        ax2 = sns.regplot(x=female_df['Age'], y=female_df['Serum_Chol'], label='female',
                color='red', ax=axes[1])

        ## Format Plot

        # Add titles
        fig.suptitle('Serum Cholesterol\n')
        ax1.set_title('Male')
        ax2.set_title('Female')
        fig.text(x=0.54, y=0.04, horizontalalignment='center', s='Age', size=11)

        # X-Axis
        ax1.set_xlim(25,84)
        ax2.set_xlim(25,84)
        ax1.set_xticks(range(35, 84, 10))
        ax2.set_xticks(range(35, 84, 10))
        ax1.set_xlabel('')
        ax2.set_xlabel('')

        # Y-Axis
        ax1.set_ylim(100, 600)
        ax2.set_ylim(100, 600)
        ax1.set_yticks(range(0, 601, 100))
        ax2.set_yticks(range(0, 601, 100))
        ax1.set_ylabel('Serum Cholesterol')
        ax2.set_ylabel('')
        ax2.set_yticklabels('')

        # Layout
        fig.tight_layout()
        fig.subplots_adjust(top=0.85, bottom=0.15, left=0.12)

        # Save image
        plt.savefig(IMG_PATH + 'graph5.png')


    # Fasting Blood Sugar > 120
    def create_plot_6(self,db):

        # Get data
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex, Fast_Blood_Sugar FROM Heart;', conn)

        # Divide data into male and female

        sns.set_style('darkgrid')

        male_df = df.query('Sex == 1')
        male_df = male_df.drop('Sex', axis=1)

        female_df = df.query('Sex == 0')
        female_df = female_df.drop('Sex', axis=1)

        # Aggregate the number of true/false, grouped by age bracket

        male_df['count'] = 1
        male_df[''] = pd.cut(male_df.Age, [20,30,40,50,60,70,80])
        male_df = male_df.pivot_table('count', index='', columns='Fast_Blood_Sugar', aggfunc='sum')

        female_df['count'] = 1
        female_df[''] = pd.cut(female_df.Age, [20,30,40,50,60,70,80])
        female_df = female_df.pivot_table('count', index='', columns='Fast_Blood_Sugar', aggfunc='sum')

        # Generate the subplots
        fig, axes = plt.subplots(nrows=1, ncols=2, )
        ax1 = male_df.plot.bar(stacked=True, ax=axes[0], rot=45, legend=False)
        ax2 = female_df.plot.bar(stacked=True, ax=axes[1], rot=45, legend=False)

        ## Format Plot

        # Titles
        fig.suptitle('Fast Blood Sugar > 120 mg/dl', x=0.45)
        ax1.set_title('Male')
        ax2.set_title('Female')

        # Legend
        fig.legend(labels =['False', 'True'], loc = (0.83, 0.7))

        # Axes labels
        ax1.set_yticks(range(0, 101, 10))
        ax2.set_yticks(range(0, 101, 10))
        ax2.set_yticklabels([])
        ax1.set_xticklabels(["21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "> 80"])
        ax2.set_xticklabels(["21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "> 80"])
        fig.text(x=0.01, y=0.5, verticalalignment='center',s='Frequency', rotation=90, size=11)
        fig.text(x=0.45, y=0.05, horizontalalignment='center', s='Age Group', size=11)

        # Layout
        fig.tight_layout()
        fig.subplots_adjust(top=0.85, left=0.1, bottom=0.2, right=0.8)

        # Save image
        plt.savefig(IMG_PATH + 'graph6.png')


    # Resting ECG
    def create_plot_7(self,db):

        # Get data
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex, Resting_ECG FROM Heart;', conn)

        ## Plot histograms male and female by type

        sns.set_style("darkgrid")               # set the background of the charts
        bins = [20,30,40,50,60,70,80]           # group age by 10's
        palette = {'color': ['b', 'r']}         # separate colours for male/female

        g = sns.FacetGrid(df, row='Sex', col='Resting_ECG', hue='Sex',
                margin_titles=True, hue_kws=palette)
        g.map(sns.distplot, 'Age', bins=bins, kde=False,
                hist_kws=dict(edgecolor="black", linewidth=1));

        ## Format Plot

        # Axis Limits
        g.axes[1,0].set_ylim(0,50)
        g.axes[1,0].set_xlim(20,80)
        g.axes[1,0].set_xticks(range(20, 80, 10))
        g.axes[1,0].set_yticks(range(0, 55, 10))

        # Titles
        plt.subplots_adjust(top=0.85, left=0.07, bottom=0.15)
        g.fig.suptitle("Resting Electrocardiographic Results\n")
        g.axes[0,0].set_title("Normal")
        g.axes[0,1].set_title("ST-T Wave Abnormality")
        g.axes[0,2].set_title("Left Ventricular Hypertrophy")

        # Overall Labels
        g.set_axis_labels('', '')
        g.fig.text(x=0.01, y=0.5, verticalalignment='center',s='Frequency', rotation=90, size=11)
        g.fig.text(x=0.5, y=0.01, horizontalalignment='center', s='Age', size=11)
        gender_labels =  ["","","Male","","","Female"]
        for i, ax in enumerate(g.axes.flat):
            plt.setp(ax.texts, text=gender_labels[i], size=11)
        g.set_xticklabels(rotation=45, labels=["20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "> 80"])

        # Save image
        plt.savefig(IMG_PATH + 'graph7.png')


    # Max Heart Rate
    def create_plot_8(self,db):

        # Get data
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex, Max_Heart_Rate FROM Heart;', conn)

        # Divide into male and female
        male_df = df.query('Sex == 1')
        female_df = df.query('Sex == 0')

        # Scatter plots
        fig, axes = plt.subplots(nrows=1, ncols=2)
        ax1 = sns.regplot(x=male_df['Age'], y=male_df['Max_Heart_Rate'], label='male', ax=axes[0])
        ax2 = sns.regplot(x=female_df['Age'], y=female_df['Max_Heart_Rate'], label='female',
                color='red', ax=axes[1])

        # Format Plot

        # Add titles
        fig.suptitle('Max Heart Rate\n')
        ax1.set_title('Male')
        ax2.set_title('Female')
        fig.text(x=0.54, y=0.04, horizontalalignment='center', s='Age', size=11)

        # X-Axis
        ax1.set_xlim(25,84)
        ax2.set_xlim(25,84)
        ax1.set_xticks(range(35, 84, 10))
        ax2.set_xticks(range(35, 84, 10))
        ax1.set_xlabel('')
        ax2.set_xlabel('')

        # Y-Axis
        ax1.set_ylim(65,219)
        ax2.set_ylim(65,219)
        ax1.set_ylabel('Max Heart Rate')
        ax2.set_ylabel('')
        ax2.set_yticklabels('')

        # Layout
        fig.tight_layout()
        fig.subplots_adjust(top=0.85, bottom=0.13)

        # Save image
        plt.savefig(IMG_PATH + 'graph8.png')

# Oldpeak
    def create_plot_10(self,db):

        # Get data
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex, Oldpeak FROM Heart;', conn)

        # Divide into male and female
        male_df = df.query('Sex == 1')
        female_df = df.query('Sex == 0')

        # Scatter plots
        fig, axes = plt.subplots(nrows=1, ncols=2)
        ax1 = sns.regplot(x=male_df['Age'], y=male_df['Oldpeak'], label='male', ax=axes[0])
        ax2 = sns.regplot(x=female_df['Age'], y=female_df['Oldpeak'], label='female',
                color='red', ax=axes[1])

        # Format Plot

        # Add titles
        fig.suptitle('Oldpeak\n')
        ax1.set_title('Male')
        ax2.set_title('Female')
        fig.text(x=0.54, y=0.04, horizontalalignment='center', s='Age', size=11)

        # X-Axis
        ax1.set_xlim(25,84)
        ax2.set_xlim(25,84)
        ax1.set_xticks(range(35, 84, 10))
        ax2.set_xticks(range(35, 84, 10))
        ax1.set_xlabel('')
        ax2.set_xlabel('')

        # Y-Axis
        ax1.set_ylim(65,219)
        ax2.set_ylim(65,219)
        ax1.set_ylabel('Oldpeak')
        ax2.set_ylabel('')
        ax2.set_yticklabels('')

        # Layout
        fig.tight_layout()
        fig.subplots_adjust(top=0.85, bottom=0.13)

        # Save image
        plt.savefig(IMG_PATH + 'graph10.png')


    # Exercice Induced Angina
    def create_plot_9(self,db):

        # Get data
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex, EI_Angina FROM Heart;', conn)

        # Divide data into male and female

        sns.set_style('darkgrid')

        male_df = df.query('Sex == 1')
        male_df = male_df.drop('Sex', axis=1)

        female_df = df.query('Sex == 0')
        female_df = female_df.drop('Sex', axis=1)

        # Aggregate the number of true/false, grouped by age bracket

        male_df['count'] = 1
        male_df[''] = pd.cut(male_df.Age, [20,30,40,50,60,70,80])
        male_df = male_df.pivot_table('count', index='', columns='EI_Angina', aggfunc='sum')

        female_df['count'] = 1
        female_df[''] = pd.cut(female_df.Age, [20,30,40,50,60,70,80])
        female_df = female_df.pivot_table('count', index='', columns='EI_Angina', aggfunc='sum')

        # Generate the subplots
        fig, axes = plt.subplots(nrows=1, ncols=2, )
        ax1 = male_df.plot.bar(stacked=True, ax=axes[0], rot=45, legend=False)
        ax2 = female_df.plot.bar(stacked=True, ax=axes[1], rot=45, legend=False)

        ## Format Plot

        # Titles
        fig.suptitle('Exercise Induced Angina', x=0.45)
        ax1.set_title('Male')
        ax2.set_title('Female')

        # Legend
        fig.legend(labels =['False', 'True'], loc = (0.83, 0.7))

        # Axes labels
        ax1.set_yticks(range(0, 101, 10))
        ax2.set_yticks(range(0, 101, 10))
        ax2.set_yticklabels([])
        ax1.set_xticklabels(["21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "> 80"])
        ax2.set_xticklabels(["31-40", "41-50", "51-60", "61-70", "71-80", "> 80"])
        fig.text(x=0.01, y=0.5, verticalalignment='center',s='Frequency', rotation=90, size=11)
        fig.text(x=0.45, y=0.05, horizontalalignment='center', s='Age Group', size=11)

        # Layout
        fig.tight_layout()
        fig.subplots_adjust(top=0.85, left=0.1, bottom=0.2, right=0.8)

        # Save image
        plt.savefig(IMG_PATH + 'graph9.png')

    # Oldpeak
    def create_plot_10(self,db):

        # Get data
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex, Oldpeak FROM Heart;', conn)

        # Divide into male and female
        male_df = df.query('Sex == 1')
        female_df = df.query('Sex == 0')

        # Scatter plots
        fig, axes = plt.subplots(nrows=1, ncols=2)
        ax1 = sns.regplot(x=male_df['Age'], y=male_df['Oldpeak'], label='male', ax=axes[0])
        ax2 = sns.regplot(x=female_df['Age'], y=female_df['Oldpeak'], label='female',
                color='red', ax=axes[1])

        # Format Plot

        # Add titles
        fig.suptitle('Oldpeak\n')
        ax1.set_title('Male')
        ax2.set_title('Female')
        fig.text(x=0.54, y=0.04, horizontalalignment='center', s='Age', size=11)

        # X-Axis
        ax1.set_xlim(25,84)
        ax2.set_xlim(25,84)
        ax1.set_xticks(range(35, 84, 10))
        ax2.set_xticks(range(35, 84, 10))
        ax1.set_xlabel('')
        ax2.set_xlabel('')

        # Y-Axis
        ax1.set_ylim(0.8)
        ax2.set_ylim(0.8)
        ax1.set_ylabel('Oldpeak')
        ax2.set_ylabel('')
        ax2.set_yticklabels('')

        # Layout
        fig.tight_layout()
        fig.subplots_adjust(top=0.85, bottom=0.13)

        # Save image
        plt.savefig(IMG_PATH + 'graph10.png')




    # Slope
    def create_plot_11(self,db):

        # Get data
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex, Slope FROM Heart;', conn)

        # Divide data into male and female

        sns.set_style('darkgrid')

        male_df = df.query('Sex == 1')
        male_df = male_df.drop('Sex', axis=1)

        female_df = df.query('Sex == 0')
        female_df = female_df.drop('Sex', axis=1)

        # Aggregate the number of true/false, grouped by age bracket

        male_df['count'] = 1
        male_df[''] = pd.cut(male_df.Age, [20,30,40,50,60,70,80])
        male_df = male_df.pivot_table('count', index='', columns='Slope', aggfunc='sum')

        female_df['count'] = 1
        female_df[''] = pd.cut(female_df.Age, [20,30,40,50,60,70,80])
        female_df = female_df.pivot_table('count', index='', columns='Slope', aggfunc='sum')

        # Generate the subplots
        fig, axes = plt.subplots(nrows=1, ncols=2, )
        ax1 = male_df.plot.bar(stacked=True, ax=axes[0], rot=45, legend=False)
        ax2 = female_df.plot.bar(stacked=True, ax=axes[1], rot=45, legend=False)

        ## Format Plot

        # Titles
        fig.suptitle('Slope', x=0.45)
        ax1.set_title('Male')
        ax2.set_title('Female')

        # Legend
        fig.legend(labels =['1', '2', '3'], loc = (0.83, 0.65), title='Slope')

        # Axes labels
        ax1.set_yticks(range(0, 101, 10))
        ax2.set_yticks(range(0, 101, 10))
        ax2.set_yticklabels([])
        ax1.set_xticklabels(["21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "> 80"])
        ax2.set_xticklabels(["31-40", "41-50", "51-60", "61-70", "71-80", "> 80"])
        fig.text(x=0.01, y=0.5, verticalalignment='center',s='Frequency', rotation=90, size=11)
        fig.text(x=0.45, y=0.05, horizontalalignment='center', s='Age Group', size=11)

        # Layout
        fig.tight_layout()
        fig.subplots_adjust(top=0.85, left=0.1, bottom=0.2, right=0.8)

        # Save image
        plt.savefig(IMG_PATH + 'graph11.png')


    # Number of Vessels Coloured by Fluroscopy
    def create_plot_12(self,db):

        # Get data
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex, Major_Vessels FROM Heart;', conn)

        ## Plot histograms male and female by type

        sns.set_style("darkgrid")               # set the background of the charts
        bins = [20,30,40,50,60,70,80]           # group age by 10's
        palette = {'color': ['b', 'r']}         # separate colours for male/female

        g = sns.FacetGrid(df, row='Sex', col='Major_Vessels', hue='Sex',
                margin_titles=True, hue_kws=palette)
        g.map(sns.distplot, 'Age', bins=bins, kde=False,
                hist_kws=dict(edgecolor="black", linewidth=1));

        ## Format Plot

        # Axis Limits
        g.axes[1,0].set_ylim(0,50)
        g.axes[1,0].set_xlim(20,80)
        g.axes[1,0].set_xticks(range(20, 80, 10))

        # Titles
        plt.subplots_adjust(top=0.85, left=0.07, bottom=0.15)
        g.fig.suptitle("Number of Vessels Coloured by Fluroscopy\n")
        g.axes[0,0].set_title("0 Vessels")
        g.axes[0,1].set_title("1 Vessel")
        g.axes[0,2].set_title("2 Vessels")
        g.axes[0,3].set_title("3 Vessels")

        # Overall Labels
        g.set_axis_labels('', '')
        g.fig.text(x=0.01, y=0.5, verticalalignment='center',s='Frequency', rotation=90, size=11)
        g.fig.text(x=0.5, y=0.01, horizontalalignment='center', s='Age', size=11)
        gender_labels =  ["","","","Male","","","","Female"]
        for i, ax in enumerate(g.axes.flat):
            plt.setp(ax.texts, text=gender_labels[i], size=11)

        g.set_xticklabels(rotation=45, labels=["20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "> 80"])

        # Save image
        plt.savefig(IMG_PATH + 'graph12.png')


    # Thalassemia
    def create_plot_13(self,db):

        # Get data
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex, Thal FROM Heart;', conn)

        ## Plot histograms male and female by type

        sns.set_style("darkgrid")               # set the background of the charts
        bins = [20,30,40,50,60,70,80]           # group age by 10's
        palette = {'color': ['b', 'r']}         # separate colours for male/female

        g = sns.FacetGrid(df, row='Sex', col='Thal', hue='Sex',
                margin_titles=True, hue_kws=palette)
        g.map(sns.distplot, 'Age', bins=bins, kde=False,
                hist_kws=dict(edgecolor="black", linewidth=1));

        ## Format Plot

        # Axis Limits
        g.axes[1,0].set_ylim(0,61)
        g.axes[1,0].set_xlim(20,80)
        g.axes[1,0].set_xticks(range(20, 80, 10))
        g.axes[1,0].set_yticks(range(0, 55, 10))

        # Titles
        plt.subplots_adjust(top=0.85, left=0.07, bottom=0.15)
        g.fig.suptitle("Thalassemia\n")
        g.axes[0,0].set_title("Normal")
        g.axes[0,1].set_title("Fixed Defect")
        g.axes[0,2].set_title("Reversible Defect")

        # Overall Labels
        g.set_axis_labels('', '')
        g.fig.text(x=0.01, y=0.5, verticalalignment='center',s='Frequency', rotation=90, size=11)
        g.fig.text(x=0.5, y=0.01, horizontalalignment='center', s='Age', size=11)
        gender_labels =  ["","","Male","","","Female"]
        for i, ax in enumerate(g.axes.flat):
            plt.setp(ax.texts, text=gender_labels[i], size=11)

        g.set_xticklabels(rotation=45, labels=["20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "> 80"])

        # Save image
        plt.savefig(IMG_PATH + 'graph13.png')