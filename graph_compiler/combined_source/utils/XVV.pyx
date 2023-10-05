import os
import sqlite3
from sqlite3 import Error


class CacheDB:
    def __init__(self):
        self.verbose = False
        ivy_root = os.getenv("IVY_ROOT", None)
        if ivy_root is None:
            raise Exception("IVY_ROOT environment variable not set")
        path = f"{ivy_root}/cache.sqlite"
        self.create_connection(path)
        self.create_cache_table()

    def create_connection(self, path):
        connection = None
        try:
            connection = sqlite3.connect(path)
            if self.verbose:
                print("Connection to SQLite DB successful")
        except Error as e:
            print(f"The error '{e}' occurred")

        self.connection = connection

    def execute_query(self, query):
        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            self.connection.commit()
            if self.verbose:
                print("Query executed successfully")
        except Error as e:
            print(f"The error '{e}' occurred")

    def execute_read_query(self, query):
        cursor = self.connection.cursor()
        result = None
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            return result
        except Error as e:
            print(f"The db read query error '{e}' occurred")

    def create_table(self, table_name, columns):
        query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        {columns}
        );
        """
        self.execute_query(query)

    def create_cache_table(self):
        columns = """
        code_loc TEXT NOT NULL,
        code_line INTEGER,
        func_def TEXT,
        args_str TEXT,
        kwargs_str TEXT,
        compile_kwargs_str TEXT,
        source_code_str TEXT,
        graph_fn_str TEXT,
        constants BLOB
        """
        self.create_table('cache', columns)

    def insert_cache(self, code_loc, code_line, func_def, args_str, kwargs_str, compile_kwargs_str, source_code_str, graph_fn_str, constants):
        query = f"""
        INSERT INTO
            cache (code_loc, code_line, func_def, args_str, kwargs_str, compile_kwargs_str, source_code_str, graph_fn_str, constants)
            VALUES
            ('{code_loc}', {code_line}, '{func_def}',  '{args_str}', '{kwargs_str}', '{compile_kwargs_str}', '{source_code_str}', '{graph_fn_str}', '{constants}')
        """
        self.execute_query(query)

    def select_all_cache(self):
        query = "SELECT * FROM cache"
        return self.execute_read_query(query)

    def select_matching_cache(self, code_loc, code_line, func_def, args_str, kwargs_str, compile_kwargs_str, source_code):
        # graph search is not used as the ObjectID is not the same
        query = f"""
        SELECT * FROM cache
        WHERE code_loc = '{code_loc}'
        AND code_line = {code_line}
        AND func_def = '{func_def}'
        AND args_str = '{args_str}'
        AND kwargs_str = '{kwargs_str}'
        AND compile_kwargs_str = '{compile_kwargs_str}'
        """
        return self.execute_read_query(query)

    def delete_cache(self, id):
        query = f"DELETE FROM cache WHERE id = {id}"
        self.execute_query(query)

    def insert_cache_and_select_all(self, code_loc, code_line, func_def, graph_fn_str, args, kwargs, source_code, constants):
        self.insert_cache(code_loc, code_line, func_def,
                          graph_fn_str, args, kwargs, source_code, constants)
        caches = self.select_all_cache()
        for cache in caches:
            print(cache)