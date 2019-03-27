import sqlite3
import pandas

# class to perform database operations

class server_database:
    # init class
    def __init__(self):
        self.database_name = 'server.db'
        try:
            conn = sqlite3.connect(self.database_name)
            self.create_table(conn)
        except sqlite3.Error as e:
            print(e)
        finally:
            conn.commit()
            conn.close()
        return
    
    # Database operations here
    def create_table(self,conn):
        c = conn.cursor()
        sql = 'CREATE TABLE IF NOT EXISTS Heart (ID INTEGER PRIMARY KEY, \
            Age Integer NOT NULL, \
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
                                                            Thal INTERGER NOT NULL, \
                                                                Target INTEGER NOT NULL)'
        c.execute(sql)
        return

    def insert_row(self, row):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        try:
            c.execute('INSERT INTO Heart VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', row)
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
    def import_data(self, data):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        rows = []
        c.executemany("INSERT INTO Heart VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        conn.commit()
        conn.close()

# Test area
db = server_database()