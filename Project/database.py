import sqlite3
import pandas

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
        #sql = 'CREATE TABLE IF NOT EXISTS <I NEED A NAME> (<I NEED FIELDS>)'
        #c.execute(sql)
