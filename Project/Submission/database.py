import sqlite3
import pandas
import numpy

# class to perform database operations

class server_database:
    # init class
    def __init__(self, db_name='static/server.db'):
        self.database_name = db_name
        try:
            conn = sqlite3.connect(self.database_name)
            self.create_table(conn)
        except sqlite3.Error as e:
            print(e)
        finally:
            conn.close()
        return
    
    # Database operations here
    def create_table(self, conn):
        c = conn.cursor()
        sql = 'CREATE TABLE IF NOT EXISTS Heart (ID INTEGER PRIMARY KEY, \
            Age INTEGER NOT NULL, \
                Sex INTEGER NOT NULL, \
                    Chest_Pain_Type INTEGER NOT NULL,\
                        RBP REAL NOT NULL, \
                            Serum_Chol REAL NOT NULL, \
                                Fast_Blood_Sugar INTEGER NOT NULL, \
                                    Resting_ECG INTEGER NOT NULL, \
                                        Max_Heart_Rate REAL NOT NULL, \
                                            EI_Angina INTEGER NOT NULL, \
                                                Oldpeak REAL NOT NULL, \
                                                    Slope INTEGER NOT NULL, \
                                                        Major_Vessels INTEGER NOT NULL, \
                                                            Thal INTEGER NOT NULL, \
                                                                Target INTEGER NOT NULL)'
        try:
            c.execute(sql)
        except sqlite3.Error as e:
            print(e)
        conn.commit()   # remove when deploy
        return

    def insert_row(self, row):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        try:
            c.execute('INSERT OR IGNORE INTO Heart VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', row)
        except sqlite3.Error as e:
            print(e)
        conn.commit()
        conn.close()
    
    def delete_row(self, condition):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        delete = 'DELETE From Heart WHERE {}'.format(condition)
        try:
            c.execute(delete)
        except sqlite3.Error as e:
            print(e)
        conn.commit()
        conn.close()

    def load_all_rows(self):
        conn = sqlite3.connect(self.database_name)
        select = "SELECT * from Heart"
        
        try:
            heartData = pandas.read_sql(select,conn)
        except sqlite3.Error as e:
            print(e)

        conn.commit()
        conn.close()
        return heartData

    def delete_table(self):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        delete = 'DROP Heart IF EXISTS'
        try:
            c.execute(delete)
        except sqlite3.Error as e:
            print(e)
        conn.commit()
        conn.close()
    
    def select_row(self, condition):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        delete = 'SELECT From Heart WHERE {}'.format(condition)
        try:
            c.execute(delete)
        except sqlite3.Error as e:
            print(e)
        conn.commit()
        conn.close()
    
    # Import/clean operations
    def import_data(self, data='static/processed.cleveland_no_headers.csv'):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()

        #get csv data
        missing_values = ['?']
        df = pandas.read_csv(data, na_values = missing_values)
        
        # clean the data
        df = df.dropna() # this will be fixed, planning on imputing data for any NaN values in real type columns
        
        #df.columns = ['Age', 'Sex', 'Chest_Pain_Type', 'RBP', 'Serum_Chol', 'Fast_Blood_Sugar', 'Resting_ECG', 'Max_Heart_Rate', 'EI_Angina', 'Oldpeak', 'Slope', 'Major_Vessels', 'Thal', 'Target']
        #desired column data types: [int, int, int, real, real, int, int, real, int, real, int, int, int, int]
        #convert appropriate columns to integers - sorry about magic numbers
        df.iloc[:,[0,1,2,5,6,8,10,11,12,13]] = df.iloc[:,[0,1,2,5,6,8,10,11,12,13]].apply(pandas.to_numeric, downcast='integer')
        
        #Column 13 - Target, Change all non-zeros to value 1 (has heart disease)
        df.iloc[:,13] = numpy.where(df.iloc[:, 13] == 0, 0, 1)

        # put tuples into rows, types converted from numpy variants to standard python types - did this to avoid sqlite3 throwing data integrity error
        # index from dataframe being used as primary key
        rows = [tuple(x) for x in df.to_records(index=True).tolist()]
        #for row in rows:
        #    print(row)
        ###
        c.executemany("INSERT OR IGNORE INTO Heart VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        conn.commit()
        conn.close()