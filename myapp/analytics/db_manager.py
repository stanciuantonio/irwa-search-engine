import sqlite3
import os

class DBManager:
    def __init__(self, db_path="analytics.db"):
        self.db_path = db_path
        self.initialize_db()

    def get_connection(self):
        """Get a database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Access columns by name
        return conn

    def initialize_db(self):
        """Initialize database tables if they don't exist"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Create sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                ip_address TEXT,
                browser TEXT,
                os_name TEXT,
                start_time TEXT
            )
        ''')

        # Create queries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS queries (
                query_id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT,
                session_id TEXT,
                algorithm TEXT,
                num_results INTEGER,
                timestamp TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        ''')

        # Create clicks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clicks (
                click_id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                session_id TEXT,
                query_id INTEGER,
                ranking_position INTEGER,
                timestamp TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id),
                FOREIGN KEY (query_id) REFERENCES queries (query_id)
            )
        ''')

        conn.commit()
        conn.close()
