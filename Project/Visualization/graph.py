'''
hack at me

I am just a header area
'''
import os, sys, inspect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CURR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURR_DIR)
sys.path.insert(0, PARENT_DIR)

from database import *

IMG_PATH = PARENT_DIR + '/Server/static/images/'    #

class graph:


    def __init__(self, attr_num, db_conn):
        self.name = 'Graph_' + str(attr_num)
        self.db = db_conn


    # Chest Pain Type
    def create_plot_3(self):

        # Get data
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex, Chest_Pain_Type, Target FROM Heart;', conn)

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
        g.axes[1,0].set_xticks(range(20, 81, 10))
        g.axes[1,0].set_yticks(range(0, 55, 10))

        # Titles
        plt.subplots_adjust(top=0.85, left=0.05)
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

        # Save image
        plt.savefig(IMG_PATH + 'graph3.png')

    # Resting Blood Pressure
    def create_plot_4(self):

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

        # X-Axis
        ax1.set_xlim(25,84)
        ax2.set_xlim(25,84)
        ax1.set_xticks(range(35, 84, 10))
        ax2.set_xticks(range(35, 84, 10))

        # Y-Axis
        ax1.set_ylim(90,210)
        ax2.set_ylim(90,210)

        # Layout
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)

        # Save image
        plt.savefig(IMG_PATH + 'graph4.png')


    # Serum Cholesterol
    def create_plot_5(self):

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

        # X-Axis
        ax1.set_xlim(25,84)
        ax2.set_xlim(25,84)
        ax1.set_xticks(range(35, 84, 10))
        ax2.set_xticks(range(35, 84, 10))

        # Y-Axis
        ax1.set_ylim(100, 580)
        ax2.set_ylim(100, 580)
        ax1.set_yticks(range(200, 600, 100))
        ax2.set_yticks(range(200, 560, 100))

        # Layout
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)

        # Save image
        plt.savefig(IMG_PATH + 'graph5.png')


    # Max Heart Rate
    def create_plot_8(self):

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

        # X-Axis
        ax1.set_xlim(25,84)
        ax2.set_xlim(25,84)
        ax1.set_xticks(range(35, 84, 10))
        ax2.set_xticks(range(35, 84, 10))

        # Y-Axis
        ax1.set_ylim(65,219)
        ax2.set_ylim(65,219)

        # Layout
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)

        # Save image
        plt.savefig(IMG_PATH + 'graph8.png')


# Main function to create each graph and save in /Server/static/images/
if __name__ == '__main__':

    # Connect to our graph instance of the database
    db = server_database('graph_db.db')

    # Create graphs and save files

    '''
    # Chest pain type
    g3 = graph(3, db)
    g3.create_plot_3()

    # Resting Blood Pressure
    g4 = graph(4, db)
    g4.create_plot_4()

    # Serum Cholestrol
    g5 = graph(5, db)
    g5.create_plot_5()

    # Fasting Blood Sugar

    # Resting ECG

    # Max Heart Rate
    g8 = graph(8, db)
    g8.create_plot_8()

    # Exercise Induced Angina

    # Oldpeak

    # Slope

    # Number Vessels Coloure by Flouroscopy

    # Thalassemia

    '''
