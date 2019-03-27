import sqlite3
import pandas

# class to perform database operations
class database:
    def __init__(self):
        self.database_name = 'database.db'
        try:
            conn = sqlite3.connect(filename)
            self.database = filename
            self.create_table(conn)
        except Error as e:
        print(e)
        finally:
            conn.commit()
            conn.close()

    def create_table(self, conn):
        c = conn.cursor()
        sql = 'CREATE TABLE IF NOT EXISTS Heart (Index INTEGER PRIMARY KEY, \
            Sex INTEGER NOT NULL, \
                Chest_Pain_Type INTEGER NOT NULL,\
                    RBP REAL NOT NULL, \
                        Serum_Chol REAL NOT NULL, \
                            Fast_Blood_Sugar INTEGER NOT NULL, \
                                Resting_ECG INTEGER NOT NULL\
                                    Max_Heart_Rate REAL NOT NULL, \
                                        EI_Angina INTEGER NOT NULL, \
                                            Oldpeak REAL NOT NULL, \
                                                Slope INTEGER NOT NULL, \
                                                    Major_Vessels INTEGER NOT NULL, \
                                                        Thal INTERGER NOT NULL, \
                                                            Target INTEGER NOT NULL)'
        c.execute(sql)
    
    def self.insert_row(self, conn):
        pass

    def self.delete_row(self, conn):
        pass
    
    def self.update_row(self, conn):
        pass
    
    def self.import_data(self, conn, data):
        pass

# Test area
db = database()