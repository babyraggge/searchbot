import sqlite3
import numpy as np
import io


def adapt_array(arr):
    return arr.tobytes()


def convert_array(text):
    return np.frombuffer(text)


class SQLiter:
    def __init__(self, database):
        sqlite3.register_adapter(np.array, adapt_array)
        sqlite3.register_converter("np_array", convert_array)

        self.connection = sqlite3.connect(database, detect_types=sqlite3.PARSE_DECLTYPES)
        self.cursor = self.connection.cursor()

    def create_table(self):
        with self.connection:
            return self.cursor.execute(
                '''
                CREATE TABLE IF NOT EXISTS code_map(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code np_array NOT NULL, 
                person TEXT NOT NULL,
                path TEXT NOT NULL)
                '''
            )

    def insert_single(self, code, name, path):
        with self.connection:
            return self.cursor.execute('INSERT INTO code_map(code, person, path) VALUES (?, ?, ?)', (code, name, path))

    def insert_batch(self, batch):
        with self.connection:
            return self.cursor.executemany('INSERT INTO code_map(code, person, path) VALUES (?, ?, ?)', batch)

    def select_all(self):
        with self.connection:
            return self.cursor.execute('SELECT * FROM code_map').fetchall()

    def select_single(self, idx):
        with self.connection:
            return self.cursor.execute('SELECT * FROM code_map WHERE id = ?', (idx,)).fetchall()[0]

    def count_rows(self):
        with self.connection:
            result = self.cursor.execute('SELECT * FROM code_map').fetchall()
            return len(result)

    def clear_base(self):
        return self.cursor.execute('DELETE FROM code_map')

    def close(self):
        self.connection.commit()
        self.connection.close()
