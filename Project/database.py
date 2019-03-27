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
    def create_table(self, conn):
        c = conn.cursor()
        sql = 'CREATE TABLE IF NOT EXISTS Heart (ID INTEGER PRIMARY KEY, \
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

    def insert_row(self, conn):
        pass
    
    def delete_row(self, conn):
        pass
    
    def delete_table(self, conn):
        pass
    
    # Import/clean operations
    def import_data(self, conn, data):
        pass

# Test area
db = database()