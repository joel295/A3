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


    # Resting Blood Pressure
    def create_plot_4(self):

        # Get data
        conn = sqlite3.connect(db.database_name)
        df = pd.read_sql_query('SELECT Age, Sex, RBP FROM Heart;', conn)

        # Divide into male and female
        male_df = df.query('Sex == 1')
        female_df = df.query('Sex == 0')

        # Scatter plot
        fig, axes = plt.subplots(nrows=1, ncols=2)
        ax1 = sns.regplot(x=male_df['Age'], y=male_df['RBP'], label='male', ax=axes[0])
        ax2 = sns.regplot(x=female_df['Age'], y=female_df['RBP'], label='female',
                color='red', ax=axes[1])

        # Add titles
        fig.suptitle('Resting Blood Pressure\n')
        ax1.set_title('Male')
        ax2.set_title('Female')

        # Format and save
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)

        # Save image
        plt.savefig(IMG_PATH + 'graph4.png')


if __name__ == '__main__':

    # Connect to our graph instance of the database
    db = server_database('graph_db.db')

    # Create graphs and save files

    # Resting Blood Pressure
    g4 = graph(1, db)
    g4.create_plot_4()
