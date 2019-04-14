'''
Visualization for Heart data attributes 3-13, with heart disease labelled in each graph
Creates graph<number>_labelled.png files
'''
import os, sys, inspect
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D

CURR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURR_DIR)
sys.path.insert(0, PARENT_DIR)

from database import *

IMG_PATH = PARENT_DIR + '/Submission/static/images/'

class graph:


    def __init__(self, attr_num, db_conn):
        self.name = 'Graph_' + str(attr_num)
        self.db = db_conn


    # Chest Pain Type - with secondary histogram
    def create_plot_3_labelled(self, db):
        
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex, Chest_Pain_Type, Target FROM Heart;', conn)

        ## Plot histograms male and female by type

        sns.set_style("darkgrid")               # set the background of the charts
        bins = [20,30,40,50,60,70,80]           # group age by 10's
        palette = {'color': ['b', 'r']}         # separate colours for male/female
        palette_dis = {'color': ['m', 'g']}
        df['Target'].values[(df['Target'].values != 0)] = 1

        df_disease = df[(df['Target']==1)] 

        male_df = df.query('Sex == 1')
        male_df_dis = male_df.query('Target == 1')

        female_df = df.query('Sex == 0')
        female_df_dis = female_df.query('Target == 1')

        male_df = male_df.drop('Sex', axis=1)
        female_df = female_df.drop('Sex', axis=1)


        df_graph = [];
        df_dis_graph = []

        for i in range(1,5):
            df_graph.append(male_df[(male_df['Chest_Pain_Type'] == i)])
            df_dis_graph.append(male_df_dis[(male_df_dis['Chest_Pain_Type'] == i)])

        for i in range(1,5):
            df_graph.append(female_df[(female_df['Chest_Pain_Type'] == i)])
            df_dis_graph.append(female_df_dis[(female_df_dis['Chest_Pain_Type'] == i)])

        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(13,6.5))
        #print (axes)
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        #plt.figure()
        count = 1
        colors = ['b', 'g']
        for a, b, c in zip(axes.flatten(), df_graph, df_dis_graph):
            if count >4:
                colors = ['r','y']
            if count%4==0:
                labels = ['Total', 'Has Heart Disease']
            else:
                labels = [None, None]
            count+=1
            sns.distplot(b['Age'],ax= a, bins=bins,color = colors[0], label = labels[0], kde=False, hist_kws=dict(edgecolor="black", linewidth=1))
            sns.distplot(c['Age'],ax= a, bins=bins,color = colors[1], label = labels[1], kde=False, hist_kws=dict(alpha=0.75, edgecolor="black", linewidth=1))
            #plt.legend()
        fig.legend()
        # Axis Limits
        for i in range(0, 8):
            fig.axes[i].set_ylim(0,50)
            fig.axes[i].set_xlim(20,80)
            fig.axes[i].set_xticks(range(20, 80, 10))
            fig.axes[i].set_yticks(range(0, 55, 10))


        ## Titles
        plt.subplots_adjust(top=0.85, left=0.07, bottom=0.15)
        fig.suptitle("Chest Pain Type\n")
        fig.axes[0].set_title("Typical Angina")
        fig.axes[1].set_title("Atypical Angina")
        fig.axes[2].set_title("Non-Anginal Pain")
        fig.axes[3].set_title("Asymptomatic")

        ## Overall Labels
        fig.text(x=0.01, y=0.5, verticalalignment='center',s='Frequency', rotation=90, size=11)
        fig.text(x=0.5, y=0.01, horizontalalignment='center', s='Age', size=11)
        gender_labels =  ["","","","Male","","","","Female","","","",]
        for i, ax in enumerate(axes.flatten()):
            plt.setp(ax.texts, text=gender_labels[i], size=11)


        plt.setp(axes, xticklabels=["20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "> 80"])
        count = 0
        for ax in fig.axes:
            ax.set_xlabel('')
            if count < 4:
                count+=1
                ax.xaxis.set_ticklabels([])
                continue
            plt.sca(ax)
            plt.xticks(rotation=45)

        rows = ['Male', 'Female']
        count = 0
        for ax in fig.axes:
            count+=1
            if count == 1 or count == 5:
                continue
            ax.yaxis.set_ticklabels([])
        
        for ax, row in zip(axes[:,3], rows):
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(row, rotation=90, size='large')
        # Save image
        plt.savefig(IMG_PATH + 'graph3_labelled.png')
        plt.close()
        

    # Resting Blood Pressure
    def create_plot_4_labelled(self, db):

        # Get data
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex, RBP, Target FROM Heart;', conn)

        # Divide into male and female
        sns.set_style('darkgrid')
        df['Target'].values[df['Target'].values != 0] = 1
        df['Heart Disease?'] = df['Target'].map({1:'Yes', 0:'No'})
        df['Sex'] = df['Sex'].map({0:'Female', 1:'Male'})
        
        plt.figure()
        g= sns.lmplot(data=df, x='Age', y='RBP',col='Sex', hue='Heart Disease?', palette=dict(Yes = 'orange', No='b'))
        g.set(xlim=(25,84), ylim=(90,210))
        plt.subplots_adjust(top=0.9, left=0.05, bottom=0.1)
        g.fig.suptitle("Resting Blood Pressure\n")
        g.axes[0,0].set_title("Male")
        g.axes[0,1].set_title("Female")
        plt.savefig(IMG_PATH + 'graph4_labelled.png')
        plt.close()


    # Serum Cholesterol
    def create_plot_5_labelled(self, db):

        # Get data
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex, Serum_Chol, Target FROM Heart;', conn)

        # Divide into male and female
        sns.set_style('darkgrid')
        df['Target'].values[df['Target'].values != 0] = 1
        df['Heart Disease?'] = df['Target'].map({1:'Yes', 0:'No'})
        df['Sex'] = df['Sex'].map({0:'Female', 1:'Male'})
        df['Serum Cholesterol (mg/dL)'] = df['Serum_Chol'];
        df.drop('Serum_Chol', axis=1)
        
        plt.figure()
        g= sns.lmplot(data=df, x='Age', y='Serum Cholesterol (mg/dL)',col='Sex', hue='Heart Disease?', palette=dict(Yes = 'orange', No='b'))
        g.set(xlim=(25,84), ylim=(100,600))
        plt.subplots_adjust(top=0.9, left=0.05, bottom=0.1)
        g.fig.suptitle("Serum Cholesterol\n")
        g.axes[0,0].set_title("Male")
        g.axes[0,1].set_title("Female")
        plt.savefig(IMG_PATH + 'graph5_labelled.png')
        plt.close()

    #fasting blood sugar
    def create_plot_6_labelled(self,db):
        # Get data
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex, Fast_Blood_Sugar, Target FROM Heart;', conn)

        # Divide data into male and female

        sns.set_style('darkgrid')

        male_df = df.query('Sex == 1')
        male_df = male_df.drop('Sex', axis=1)

        female_df = df.query('Sex == 0')
        female_df = female_df.drop('Sex', axis=1)

        male_df_fasted = male_df.query('Fast_Blood_Sugar == 1')
        male_df_unfasted = male_df.query('Fast_Blood_Sugar == 0')

        female_df_fasted = female_df.query('Fast_Blood_Sugar == 1')
        female_df_unfasted = female_df.query('Fast_Blood_Sugar == 0')

        # Aggregate the number of true/false, grouped by age bracket

        male_df_fasted['count'] = 1
        male_df_fasted[''] = pd.cut(male_df_fasted.Age, [20,30,40,50,60,70,80])
        male_df_fasted = male_df_fasted.pivot_table('count', index='', columns='Target', aggfunc='sum')

        male_df_unfasted['count'] = 1
        male_df_unfasted[''] = pd.cut(male_df_unfasted.Age, [20,30,40,50,60,70,80])
        male_df_unfasted = male_df_unfasted.pivot_table('count', index='', columns='Target', aggfunc='sum')

        female_df_fasted['count'] = 1
        female_df_fasted[''] = pd.cut(female_df_fasted.Age, [20,30,40,50,60,70,80])
        female_df_fasted = female_df_fasted.pivot_table('count', index='', columns='Target', aggfunc='sum')

        female_df_unfasted['count'] = 1
        female_df_unfasted[''] = pd.cut(female_df_unfasted.Age, [20,30,40,50,60,70,80])
        female_df_unfasted = female_df_unfasted.pivot_table('count', index='', columns='Target', aggfunc='sum')

        # Generate the subplots
        fig, axes = plt.subplots(nrows=2, ncols=2)
        ax1 = male_df_fasted.plot.bar(stacked=True, ax=axes[0,0], rot=0, legend=False)
        ax2 = female_df_fasted.plot.bar(stacked=True, ax=axes[0,1], rot=0, legend=False)
        ax3 = male_df_unfasted.plot.bar(stacked=True, ax=axes[1,0], rot=45, legend=False)
        ax4 = female_df_unfasted.plot.bar(stacked=True, ax=axes[1,1], rot=45, legend=False)

        ## Format Plot

        # Titles
        fig.suptitle('Fasted Blood Sugar > 120 mg/dL')
        ax1.set_title('Male')
        ax2.set_title('Female')
        

        # Legend
        fig.legend(title='Has Heart Disease?',labels =['No', 'Yes'] )#loc = (0.83, 0.7))

        # Axes labels
        ax1.set_yticks(range(0, 20, 2))
        ax2.set_yticks(range(0, 20, 2))
        ax2.set_yticklabels([])
        ax2.set_ylabel('True')
        ax2.yaxis.set_label_position("right")

        ax1.set_xticklabels(["21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "> 80"])
        ax2.set_xticklabels(["21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "> 80"])
        ax3.set_yticks(range(0, 101, 10))
        ax4.set_yticks(range(0, 101, 10))
        ax4.set_yticklabels([])
        ax3.set_xticklabels(["21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "> 80"])
        ax4.set_xticklabels(["21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "> 80"])
        ax4.set_ylabel('False')
        ax4.yaxis.set_label_position("right")
        #fig.text(x=0.01, y=0.5, verticalalignment='center',s='Frequency', rotation=90, size=11)
        #fig.text(x=0.45, y=0.05, horizontalalignment='center', s='Age Group', size=11)

        # Layout
        #fig.tight_layout()
        #fig.subplots_adjust(top=0.85, left=0.1, bottom=0.2, right=0.8)

        # Save image
        plt.savefig(IMG_PATH + 'graph6_labelled.png')
        plt.close()


    # Resting ECG
    def create_plot_7_labelled(self, db):

        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex, Resting_ECG, Target FROM Heart;', conn)

        ## Plot histograms male and female by type

        sns.set_style("darkgrid")               # set the background of the charts
        bins = [20,30,40,50,60,70,80]           # group age by 10's
        palette = {'color': ['b', 'r']}         # separate colours for male/female
        palette_dis = {'color': ['m', 'g']}
        df['Target'].values[(df['Target'].values != 0)] = 1

        df_disease = df[(df['Target']==1)] 

        male_df = df.query('Sex == 1')
        male_df_dis = male_df.query('Target == 1')

        female_df = df.query('Sex == 0')
        female_df_dis = female_df.query('Target == 1')

        male_df = male_df.drop('Sex', axis=1)
        female_df = female_df.drop('Sex', axis=1)


        df_graph = [];
        df_dis_graph = []

        for i in range(0,3):
            df_graph.append(male_df[(male_df['Resting_ECG'] == i)])
            df_dis_graph.append(male_df_dis[(male_df_dis['Resting_ECG'] == i)])

        for i in range(0,3):
            df_graph.append(female_df[(female_df['Resting_ECG'] == i)])
            df_dis_graph.append(female_df_dis[(female_df_dis['Resting_ECG'] == i)])

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(13,6.5))
        #print (axes)
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        #plt.figure()
        count = 1
        colors = ['b', 'g']
        for a, b, c in zip(axes.flatten(), df_graph, df_dis_graph):
            if count>3:
                colors = ['r','y']
            if count%3==0:
                labels = ['Total', 'Has Heart Disease']
            else:
                labels = [None, None]
            count+=1
            sns.distplot(b['Age'],ax= a, bins=bins,color = colors[0], label = labels[0], kde=False, hist_kws=dict(edgecolor="black", linewidth=1))
            sns.distplot(c['Age'],ax= a, bins=bins,color = colors[1], label = labels[1], kde=False, hist_kws=dict(alpha=0.75, edgecolor="black", linewidth=1))
            #plt.legend()
        fig.legend()
        # Axis Limits
        for i in range(0, 6):
            fig.axes[i].set_ylim(0,50)
            fig.axes[i].set_xlim(20,80)
            fig.axes[i].set_xticks(range(20, 80, 10))
            fig.axes[i].set_yticks(range(0, 55, 10))


        ## Titles
        plt.subplots_adjust(top=0.85, left=0.07, bottom=0.15)
        fig.suptitle("Resting Electrocardiographic Results\n")
        fig.axes[0].set_title("Normal")
        fig.axes[1].set_title("ST-T Wave Abnormality")
        fig.axes[2].set_title("Left Ventricular Hypertrophy")

        ## Overall Labels
        fig.text(x=0.01, y=0.5, verticalalignment='center',s='Frequency', rotation=90, size=11)
        fig.text(x=0.5, y=0.01, horizontalalignment='center', s='Age', size=11)
        gender_labels =  ["","","Male","","","Female"]
        for i, ax in enumerate(axes.flatten()):
            plt.setp(ax.texts, text=gender_labels[i], size=11)


        plt.setp(axes, xticklabels=["20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "> 80"])
        count = 0
        for ax in fig.axes:
            ax.set_xlabel('')
            if count < 3:
                count+=1
                ax.xaxis.set_ticklabels([])
                continue
            plt.sca(ax)
            plt.xticks(rotation=45)

        rows = ['Male', 'Female']
        count = 0
        for ax in fig.axes:
            count+=1
            if count == 1 or count == 4:
                continue
            ax.yaxis.set_ticklabels([])
        
        for ax, row in zip(axes[:,2], rows):
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(row, rotation=90, size='large')
        # Save image

        # Save image
        plt.savefig(IMG_PATH + 'graph7_labelled.png')
        plt.close()


    # Max Heart Rate
    def create_plot_8_labelled(self, db):

        # Get data
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex, Max_Heart_Rate, Target FROM Heart;', conn)

        # Divide into male and female
        sns.set_style('darkgrid')
        df['Target'].values[df['Target'].values != 0] = 1
        df['Heart Disease?'] = df['Target'].map({1:'Yes', 0:'No'})
        df['Sex'] = df['Sex'].map({0:'Female', 1:'Male'})
        df['Maximum Heart Rate (bpm)'] = df['Max_Heart_Rate'];
        df.drop('Max_Heart_Rate', axis=1)
        
        plt.figure()
        g= sns.lmplot(data=df, x='Age', y='Maximum Heart Rate (bpm)',col='Sex', hue='Heart Disease?', palette=dict(Yes = 'orange', No='b'))
        g.set(xlim=(25,84), ylim=(60,220))
        plt.subplots_adjust(top=0.9, left=0.05, bottom=0.1)
        g.fig.suptitle("Maximum Heart Rate Achieved\n")
        g.axes[0,0].set_title("Male")
        g.axes[0,1].set_title("Female")
        # Save image
        plt.savefig(IMG_PATH + 'graph8_labelled.png')
        plt.close()


    # Exercice Induced Angina
    def create_plot_9_labelled(self,db):

        # Get data
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex, EI_Angina, Target FROM Heart;', conn)
        # Divide data into male and female

        sns.set_style('darkgrid')

        male_df = df.query('Sex == 1')
        male_df = male_df.drop('Sex', axis=1)

        female_df = df.query('Sex == 0')
        female_df = female_df.drop('Sex', axis=1)

        male_df_EI = male_df.query('EI_Angina == 1')
        male_df_NoEI = male_df.query('EI_Angina == 0')

        female_df_EI = female_df.query('EI_Angina == 1')
        female_df_NoEI = female_df.query('EI_Angina == 0')

        # Aggregate the number of true/false, grouped by age bracket

        male_df_EI['count'] = 1
        male_df_EI[''] = pd.cut(male_df_EI.Age, [20,30,40,50,60,70,80])
        male_df_EI = male_df_EI.pivot_table('count', index='', columns='Target', aggfunc='sum')

        male_df_NoEI['count'] = 1
        male_df_NoEI[''] = pd.cut(male_df_NoEI.Age, [20,30,40,50,60,70,80])
        male_df_NoEI = male_df_NoEI.pivot_table('count', index='', columns='Target', aggfunc='sum')

        female_df_EI['count'] = 1
        female_df_EI[''] = pd.cut(female_df_EI.Age, [20,30,40,50,60,70,80])
        female_df_EI = female_df_EI.pivot_table('count', index='', columns='Target', aggfunc='sum')

        female_df_NoEI['count'] = 1
        female_df_NoEI[''] = pd.cut(female_df_NoEI.Age, [20,30,40,50,60,70,80])
        female_df_NoEI = female_df_NoEI.pivot_table('count', index='', columns='Target', aggfunc='sum')

        # Generate the subplots
        fig, axes = plt.subplots(nrows=2, ncols=2)
        ax1 = male_df_EI.plot.bar(stacked=True, ax=axes[0,0], rot=0, legend=False)
        ax2 = female_df_EI.plot.bar(stacked=True, ax=axes[0,1], rot=0, legend=False)
        ax3 = male_df_NoEI.plot.bar(stacked=True, ax=axes[1,0], rot=45, legend=False)
        ax4 = female_df_NoEI.plot.bar(stacked=True, ax=axes[1,1], rot=45, legend=False)

        ## Format Plot

        # Titles
        fig.suptitle('Exercise Induced Angina')
        ax1.set_title('Male')
        ax2.set_title('Female')
        

        # Legend
        fig.legend(title='Has Heart Disease?',labels =['No', 'Yes'])#loc = (0.93, 0.7) )#loc = (0.83, 0.7))

        # Axes labels
        ax1.set_yticks(range(0, 60, 5))
        ax2.set_yticks(range(0, 60, 5))
        ax2.set_yticklabels([])
        ax2.set_ylabel('True')
        ax2.yaxis.set_label_position("right")

        ax1.set_xticklabels(["21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "> 80"])
        ax2.set_xticklabels(["21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "> 80"])
        ax3.set_yticks(range(0, 60, 5))
        ax4.set_yticks(range(0, 60, 5))
        ax4.set_yticklabels([])
        ax3.set_xticklabels(["21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "> 80"])
        ax4.set_xticklabels(["21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "> 80"])
        ax4.set_ylabel('False')
        ax4.yaxis.set_label_position("right")
        #fig.text(x=0.01, y=0.5, verticalalignment='center',s='Frequency', rotation=90, size=11)
        #fig.text(x=0.45, y=0.05, horizontalalignment='center', s='Age Group', size=11)

        # Layout
        #fig.tight_layout()
        #fig.subplots_adjust(top=0.85, left=0.1, bottom=0.2, right=0.8)

        # Save image
        plt.savefig(IMG_PATH + 'graph9_labelled.png')
        plt.close()


# Oldpeak
    def create_plot_10_labelled(self, db):

        # Get data
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex, Oldpeak, Target FROM Heart;', conn)

        # Divide into male and female
        sns.set_style('darkgrid')
        df['Target'].values[df['Target'].values != 0] = 1
        df['Heart Disease?'] = df['Target'].map({1:'Yes', 0:'No'})
        df['Sex'] = df['Sex'].map({0:'Female', 1:'Male'})
        
        
        plt.figure()
        g= sns.lmplot(data=df, x='Age', y='Oldpeak',col='Sex', hue='Heart Disease?', palette=dict(Yes = 'orange', No='b'))
        g.set(xlim=(25,84), ylim=(0,6))
        plt.subplots_adjust(top=0.9, left=0.05, bottom=0.1)
        g.fig.suptitle("Oldpeak (ST depression induced by exercise relative to rest)\n")
        g.axes[0,0].set_title("Male")
        g.axes[0,1].set_title("Female")

        # Save image
        plt.savefig(IMG_PATH + 'graph10_labelled.png')
        plt.close()


    # Slope
    #def create_plot_11(self):

        

    # Number of Vessels Coloured by Fluroscopy
    def create_plot_12_labelled(self, db):

        # Get data
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex, Major_Vessels, Target FROM Heart;', conn)

        ## Plot histograms male and female by type

        sns.set_style("darkgrid")               # set the background of the charts
        bins = [20,30,40,50,60,70,80]           # group age by 10's
        palette = {'color': ['b', 'r']}         # separate colours for male/female

        df['Target'].values[(df['Target'].values != 0)] = 1

        df_disease = df[(df['Target']==1)] 

        male_df = df.query('Sex == 1')
        male_df_dis = male_df.query('Target == 1')

        female_df = df.query('Sex == 0')
        female_df_dis = female_df.query('Target == 1')

        male_df = male_df.drop('Sex', axis=1)
        female_df = female_df.drop('Sex', axis=1)


        df_graph = [];
        df_dis_graph = []

        for i in range(0,4):
            df_graph.append(male_df[(male_df['Major_Vessels'] == i)])
            df_dis_graph.append(male_df_dis[(male_df_dis['Major_Vessels'] == i)])

        for i in range(0,4):
            df_graph.append(female_df[(female_df['Major_Vessels'] == i)])
            df_dis_graph.append(female_df_dis[(female_df_dis['Major_Vessels'] == i)])

        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(13,6.5))
        #print (axes)
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        #plt.figure()
        count = 1
        colors = ['b', 'g']
        for a, b, c in zip(axes.flatten(), df_graph, df_dis_graph):
            if count >4:
                colors = ['r','y']
            if count%4==0:
                labels = ['Total', 'Has Heart Disease']
            else:
                labels = [None, None]
            count+=1
            sns.distplot(b['Age'],ax= a, bins=bins,color = colors[0], label = labels[0], kde=False, hist_kws=dict(edgecolor="black", linewidth=1))
            sns.distplot(c['Age'],ax= a, bins=bins,color = colors[1], label = labels[1], kde=False, hist_kws=dict(alpha=0.75, edgecolor="black", linewidth=1))
            #plt.legend()
        fig.legend()
        # Axis Limits
        for i in range(0, 8):
            fig.axes[i].set_ylim(0,50)
            fig.axes[i].set_xlim(20,80)
            fig.axes[i].set_xticks(range(20, 80, 10))
            fig.axes[i].set_yticks(range(0, 55, 10))


        ## Titles
        plt.subplots_adjust(top=0.85, left=0.07, bottom=0.15)
        fig.suptitle("Number of Vessels Coloured by Fluroscopy\n")
        fig.axes[0].set_title("0 Vessels")
        fig.axes[1].set_title("1 Vessel")
        fig.axes[2].set_title("2 Vessels")
        fig.axes[3].set_title("3 Vessels")

        ## Overall Labels
        fig.text(x=0.01, y=0.5, verticalalignment='center',s='Frequency', rotation=90, size=11)
        fig.text(x=0.5, y=0.01, horizontalalignment='center', s='Age', size=11)
        gender_labels =  ["","","","Male","","","","Female","","","",]
        for i, ax in enumerate(axes.flatten()):
            plt.setp(ax.texts, text=gender_labels[i], size=11)


        plt.setp(axes, xticklabels=["20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "> 80"])
        count = 0
        for ax in fig.axes:
            ax.set_xlabel('')
            if count < 4:
                count+=1
                ax.xaxis.set_ticklabels([])
                continue
            plt.sca(ax)
            plt.xticks(rotation=45)

        rows = ['Male', 'Female']
        count = 0
        for ax in fig.axes:
            count+=1
            if count == 1 or count == 5:
                continue
            ax.yaxis.set_ticklabels([])
        
        for ax, row in zip(axes[:,3], rows):
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(row, rotation=90, size='large')
       
        # Save image
        plt.savefig(IMG_PATH + 'graph12_labelled.png')
        plt.close()


    # Thalassemia
    def create_plot_13_labelled(self, db):

        # Get data
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex, Thal, Target FROM Heart;', conn)

        ## Plot histograms male and female by type

        sns.set_style("darkgrid")               # set the background of the charts
        bins = [20,30,40,50,60,70,80]           # group age by 10's
        palette = {'color': ['b', 'r']}         # separate colours for male/female

        df['Target'].values[(df['Target'].values != 0)] = 1
        df['Thal'].values[(df['Thal'].values == 7)] = 9

        df_disease = df[(df['Target']==1)] 

        male_df = df.query('Sex == 1')
        male_df_dis = male_df.query('Target == 1')

        female_df = df.query('Sex == 0')
        female_df_dis = female_df.query('Target == 1')

        male_df = male_df.drop('Sex', axis=1)
        female_df = female_df.drop('Sex', axis=1)


        df_graph = [];
        df_dis_graph = []

        for i in range(0,3):
            df_graph.append(male_df[(male_df['Thal'] == (3*i+3))])
            df_dis_graph.append(male_df_dis[(male_df_dis['Thal'] == (3*i+3))])

        for i in range(0,3):
            df_graph.append(female_df[(female_df['Thal'] == (3*i+3))])
            df_dis_graph.append(female_df_dis[(female_df_dis['Thal'] == (3*i+3))])

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(13,6.5))
        #print (axes)
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        #plt.figure()
        count = 1
        colors = ['b', 'g']
        for a, b, c in zip(axes.flatten(), df_graph, df_dis_graph):
            if count>3:
                colors = ['r','y']
            if count%3==0:
                labels = ['Total', 'Has Heart Disease']
            else:
                labels = [None, None]
            count+=1
            sns.distplot(b['Age'],ax= a, bins=bins,color = colors[0], label = labels[0], kde=False, hist_kws=dict(edgecolor="black", linewidth=1))
            sns.distplot(c['Age'],ax= a, bins=bins,color = colors[1], label = labels[1], kde=False, hist_kws=dict(alpha=0.75, edgecolor="black", linewidth=1))
            #plt.legend()
        fig.legend()
        # Axis Limits
        for i in range(0, 6):
            fig.axes[i].set_ylim(0,50)
            fig.axes[i].set_xlim(20,80)
            fig.axes[i].set_xticks(range(20, 80, 10))
            fig.axes[i].set_yticks(range(0, 55, 10))


        ## Titles
        plt.subplots_adjust(top=0.85, left=0.07, bottom=0.15)
        fig.suptitle("Thalassemia\n")
        fig.axes[0].set_title("Normal")
        fig.axes[1].set_title("Fixed Defect")
        fig.axes[2].set_title("Reversible Defect")

        ## Overall Labels
        fig.text(x=0.01, y=0.5, verticalalignment='center',s='Frequency', rotation=90, size=11)
        fig.text(x=0.5, y=0.01, horizontalalignment='center', s='Age', size=11)
        gender_labels =  ["","","Male","","","Female"]
        for i, ax in enumerate(axes.flatten()):
            plt.setp(ax.texts, text=gender_labels[i], size=11)


        plt.setp(axes, xticklabels=["20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "> 80"])
        count = 0
        for ax in fig.axes:
            ax.set_xlabel('')
            if count < 3:
                count+=1
                ax.xaxis.set_ticklabels([])
                continue
            plt.sca(ax)
            plt.xticks(rotation=45)

        rows = ['Male', 'Female']
        count = 0
        for ax in fig.axes:
            count+=1
            if count == 1 or count == 4:
                continue
            ax.yaxis.set_ticklabels([])
        
        for ax, row in zip(axes[:,2], rows):
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(row, rotation=90, size='large')

        #save image
        plt.savefig(IMG_PATH + 'graph13_labelled.png')
        plt.close()
        #pass


    def kMeansClustering(self):
    
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Max_Heart_Rate, Target FROM Heart;', conn)
        df['Target'].values[df['Target'].values != 0] = 1

        # Split the data into test and train parts
        df_no_label = df.drop('Target', axis=1)
        # Fit a k-means estimator
        estimator = KMeans(n_clusters=2)
        estimator.fit(df_no_label)
        # Clusters are given in the labels_ attribute
        labels = estimator.labels_
        df['cluster'] = pd.Series(labels, index=df.index)
    
        #print(labels)
        # divide the dataset into three dataframes based on the species
        cluster_0_df = df.query('cluster == 0')
        cluster_1_df = df.query('cluster == 1')
    
        fig, axes = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(18.5, 10.5)
        fig.tight_layout()
    
        ax = sns.scatterplot(x='Age', y='Max_Heart_Rate', label='Cluster-0', color='blue', ax=axes)
        ax = cluster_1_df.plot.scatter(x='Age', y='Max_Heart_Rate', label='Cluster-1', color='red', ax=ax)
        #ax = cluster_2_df.plot.scatter(x='petal_length', y='petal_width', label='Cluster-2', color='green', ax=ax)
    
        for i, label in enumerate(df['Target']):
    
            #label = label[0:1]
            ax.annotate(label, (list(df['Age'])[i], list(df['Max_Heart_Rate'])[i]), color='gray', fontSize=9,
                        horizontalalignment='left',
                        verticalalignment='bottom')
    
        plt.savefig(IMG_PATH + 'graph_kMeans_cluster.png')
        plt.close()
        pass


    def create_plot_15_labelled(self, db):
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex,Max_Heart_Rate, Serum_Chol,Target FROM Heart;', conn)
        df['Target'].values[df['Target'].values != 0] = 1
        sns.set_style("darkgrid")

        # Divide into male and female
        male_df = df.query('Sex == 1')
        female_df = df.query('Sex == 0')

        dis_df = male_df.query('Target ==1')
        dis_free_df = male_df.query('Target==0')

        fig = plt.figure(figsize=plt.figaspect(0.5))

        
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.scatter3D(dis_df['Max_Heart_Rate'], dis_df['Serum_Chol'],dis_df['Age'], c='orange', label = 'Yes');
        ax.scatter3D(dis_free_df['Max_Heart_Rate'], dis_free_df['Serum_Chol'],dis_free_df['Age'],  c='b', label='No');
        ax.set_xlabel('Max Heart Rate (bpm)')
        ax.set_ylabel('Serum Cholesterol (mg/dL)')
        ax.set_zlabel('Age')
        ax.set_title('Male')
        ax.legend(title = "Heart Disease?",loc='upper left')
        plt.subplots_adjust(top=0.9, left=0.05, bottom=0.1)
        ax.set_xlim(60,220)
        ax.set_ylim(100,600)
        ax.set_zlim(25,84)
        #ax.set_yticks(range(75, 225, 25))
        ax.set_xticks(range(75, 225, 25))
        ax.set_zticks(range(25, 80, 10))
        

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        dis_df = female_df.query('Target==1')
        dis_free_df = female_df.query('Target==0')

        ax.scatter3D(dis_df['Max_Heart_Rate'], dis_df['Serum_Chol'],dis_df['Age'], c='orange', label = 'Yes');
        ax.scatter3D(dis_free_df['Max_Heart_Rate'], dis_free_df['Serum_Chol'],dis_free_df['Age'],  c='b', label='No');
        ax.set_xlabel('Max Heart Rate (bpm)')
        ax.set_ylabel('Serum Cholesterol (mg/dL)')
        ax.set_zlabel('Age')
        ax.set_title('Female')
        ax.legend(title = "Heart Disease?",loc='upper left')
        fig.suptitle('Max Heart Rate vs Serum Cholesterol vs Age')
        ax.set_xlim(60,220)
        ax.set_ylim(100,600)
        ax.set_zlim(25,84)
        #ax.set_yticks(range(75, 225, 25))
        ax.set_xticks(range(75, 225, 25))
        ax.set_zticks(range(25, 80, 10))
        plt.savefig(IMG_PATH + 'graph15_labelled.png')
        plt.close()

    def create_plot_16_labelled(self, db):
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex, Chest_Pain_Type, Target FROM Heart;', conn)
        sns.set_style('darkgrid')
        df['Target'].values[df['Target'].values != 0] = 1
        df['Heart Disease?'] = df['Target'].map({1:'Yes', 0:'No'})
        df['Sex'] = df['Sex'].map({0:'Female', 1:'Male'})
        
        plt.figure()
        g= sns.lmplot(data=df, x='Max_Heart_Rate', y='Serum_Chol',col='Sex', hue='Heart Disease?', palette=dict(Yes = 'orange', No='b'))
        g.set(xlim=(60,220), ylim=(100,600))
        plt.subplots_adjust(top=0.9, left=0.05, bottom=0.1)
        g.fig.suptitle("Maximum Heart Rate vs Serum Cholesterol\n")
        g.axes[0,0].set_title("Male")
        g.axes[0,1].set_title("Female")
        
        plt.savefig(IMG_PATH + 'graph16_labelled.png')


    def create_plot_17_labelled(self,db):
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex,Max_Heart_Rate,Oldpeak,Target FROM Heart;', conn)
        df['Target'].values[df['Target'].values != 0] = 1
        
        sns.set_style("darkgrid")

        # Divide into male and female
        male_df = df.query('Sex == 1')
        female_df = df.query('Sex == 0')

        dis_df = male_df.query('Target ==1')
        dis_free_df = male_df.query('Target==0')

        fig = plt.figure(figsize=plt.figaspect(0.5))

        
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.scatter3D(dis_df['Max_Heart_Rate'], dis_df['Oldpeak'],dis_df['Age'], c='orange', label = 'Yes');
        ax.scatter3D(dis_free_df['Max_Heart_Rate'], dis_free_df['Oldpeak'],dis_free_df['Age'],  c='b', label='No');
        ax.set_xlabel('Max Heart Rate (bpm)')
        ax.set_ylabel('Oldpeak')
        ax.set_zlabel('Age')
        ax.set_title('Male')
        ax.legend(title = "Heart Disease?",loc='upper left')
        plt.subplots_adjust(top=0.9, left=0.05, bottom=0.1)
        ax.set_xlim(60,220)
        ax.set_ylim(0,6)
        ax.set_zlim(25,84)
        #ax.set_yticks(range(75, 225, 25))
        ax.set_xticks(range(75, 225, 25))
        ax.set_zticks(range(25, 80, 10))
        

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        dis_df = female_df.query('Target==1')
        dis_free_df = female_df.query('Target==0')

        ax.scatter3D(dis_df['Max_Heart_Rate'], dis_df['Oldpeak'],dis_df['Age'], c='orange', label = 'Yes');
        ax.scatter3D(dis_free_df['Max_Heart_Rate'], dis_free_df['Oldpeak'],dis_free_df['Age'],  c='b', label='No');
        ax.set_xlabel('Max Heart Rate (bpm)')
        ax.set_ylabel('Oldpeak')
        ax.set_zlabel('Age')
        ax.set_title('Female')
        ax.legend(title = "Heart Disease?",loc='upper left')
        fig.suptitle('Max Heart Rate vs Oldpeak vs Age')
        ax.set_xlim(60,220)
        ax.set_ylim(0,6)
        ax.set_zlim(25,84)
    
        ax.set_xticks(range(75, 225, 25))
        ax.set_zticks(range(25, 80, 10))
        plt.savefig(IMG_PATH + 'graph17_labelled.png')


    def create_plot_18_labelled(self, db):
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex, Chest_Pain_Type, Target FROM Heart;', conn)

        sns.set_style('darkgrid')
        df['Target'].values[df['Target'].values != 0] = 1
        df['Heart Disease?'] = df['Target'].map({1:'Yes', 0:'No'})
        df['Sex'] = df['Sex'].map({0:'Female', 1:'Male'})
        
        plt.figure()
        g= sns.lmplot(data=df,x='Max_Heart_Rate',y='Oldpeak',col='Sex', hue='Heart Disease?', palette=dict(Yes = 'orange', No='b'))
        g.set(xlim=(60,220), ylim=(0,6))
        plt.subplots_adjust(top=0.9, left=0.05, bottom=0.1)
        g.fig.suptitle("Maximum Heart Rate vs Oldpeak\n")
        g.axes[0,0].set_title("Male")
        g.axes[0,1].set_title("Female")
            
        plt.savefig(IMG_PATH + 'graph_18_labelled.png')
    