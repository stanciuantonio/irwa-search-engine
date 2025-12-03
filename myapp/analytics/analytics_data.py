"""
Analytics Data Module - Web Analytics for IRWA Search Engine
"""

import json
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

import altair as alt
import pandas as pd

from myapp.analytics.db_manager import DBManager


class AnalyticsData:
    """
    SQLite-backed analytics storage.
    Tracks basic metrics: clicks, queries, sessions.
    """

    def __init__(self):
        self.db = DBManager()

    def save_query_terms(self, query_text: str, session_id: str = None,
                         algorithm: str = 'tfidf', num_results: int = 0) -> int:
        """Save a search query and return query_id"""
        conn = self.db.get_connection()
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()

        cursor.execute('''
            INSERT INTO queries (query_text, session_id, algorithm, num_results, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (query_text, session_id, algorithm, num_results, timestamp))

        query_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return query_id

    def save_click(self, doc_id: str, session_id: str = None,
                   query_id: int = None, ranking_position: int = None):
        """Save a document click"""
        conn = self.db.get_connection()
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()

        cursor.execute('''
            INSERT INTO clicks (doc_id, session_id, query_id, ranking_position, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (doc_id, session_id, query_id, ranking_position, timestamp))

        conn.commit()
        conn.close()

    def get_or_create_session(self, session_id: str, ip_address: str = None,
                               browser: str = None, os_name: str = None) -> Dict:
        """Get or create a session"""
        conn = self.db.get_connection()
        cursor = conn.cursor()

        # Check if session exists
        cursor.execute('SELECT * FROM sessions WHERE session_id = ?', (session_id,))
        row = cursor.fetchone()

        if row:
            session_data = dict(row)
        else:
            # Create new session
            start_time = datetime.now().isoformat()
            cursor.execute('''
                INSERT INTO sessions (session_id, ip_address, browser, os_name, start_time)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, ip_address, browser, os_name, start_time))
            conn.commit()

            session_data = {
                'session_id': session_id,
                'ip_address': ip_address,
                'browser': browser,
                'os_name': os_name,
                'start_time': start_time
            }

        conn.close()
        return session_data

    # Summary statistics
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for dashboard"""
        conn = self.db.get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM sessions')
        total_sessions = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM queries')
        total_queries = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM clicks')
        total_clicks = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(DISTINCT doc_id) FROM clicks')
        unique_docs_clicked = cursor.fetchone()[0]

        conn.close()

        return {
            'total_sessions': total_sessions,
            'total_queries': total_queries,
            'total_clicks': total_clicks,
            'unique_docs_clicked': unique_docs_clicked
        }

    def get_top_queries(self, limit: int = 10) -> List[Dict]:
        """Get most frequent queries"""
        conn = self.db.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT query_text as query, COUNT(*) as count
            FROM queries
            GROUP BY query_text
            ORDER BY count DESC
            LIMIT ?
        ''', (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_algorithm_distribution(self) -> Dict[str, int]:
        """Get distribution of algorithms used"""
        conn = self.db.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT algorithm, COUNT(*) as count
            FROM queries
            GROUP BY algorithm
        ''')

        rows = cursor.fetchall()
        conn.close()

        return {row['algorithm']: row['count'] for row in rows}

    @property
    def fact_clicks(self) -> Dict[str, int]:
        """
        Property to maintain compatibility with existing code that accesses fact_clicks directly.
        Returns a dictionary of doc_id -> click_count
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT doc_id, COUNT(*) as count FROM clicks GROUP BY doc_id')
        rows = cursor.fetchall()
        conn.close()

        return {row['doc_id']: row['count'] for row in rows}

    # Visualization Methods
    def plot_number_of_views(self):
        """Bar chart of document views"""
        conn = self.db.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT doc_id, COUNT(*) as count
            FROM clicks
            GROUP BY doc_id
            ORDER BY count DESC
            LIMIT 10
        ''')
        rows = cursor.fetchall()
        conn.close()

        data = [{'Document ID': row['doc_id'][:12] + '...' if len(row['doc_id']) > 12 else row['doc_id'],
                 'Views': row['count']}
                for row in rows]

        if not data:
            data = [{'Document ID': 'No data yet', 'Views': 0}]

        df = pd.DataFrame(data)
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Document ID:N', sort='-y'),
            y='Views:Q',
            color=alt.value('#4CAF50')
        ).properties(
            title='Top 10 Most Viewed Documents',
            width=400,
            height=250
        )
        return chart.to_html()

    def plot_top_queries(self):
        """Bar chart of top queries"""
        top_queries = self.get_top_queries(10)
        if not top_queries:
            top_queries = [{'query': 'No queries yet', 'count': 0}]

        df = pd.DataFrame(top_queries)
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('query:N', sort='-y', title='Search Query'),
            y=alt.Y('count:Q', title='Frequency'),
            color=alt.value('#2196F3')
        ).properties(
            title='Top 10 Search Queries',
            width=400,
            height=250
        )
        return chart.to_html()

    def plot_algorithm_distribution(self):
        """Pie chart of algorithm usage"""
        algo_dist = self.get_algorithm_distribution()
        if not algo_dist:
            algo_dist = {'No data': 1}

        data = [{'Algorithm': k.upper(), 'Count': v} for k, v in algo_dist.items()]
        df = pd.DataFrame(data)
        chart = alt.Chart(df).mark_arc().encode(
            theta='Count:Q',
            color='Algorithm:N',
            tooltip=['Algorithm', 'Count']
        ).properties(
            title='Algorithm Usage Distribution',
            width=250,
            height=250
        )
        return chart.to_html()


# Legacy support
class ClickedDoc:
    def __init__(self, doc_id, description, counter):
        self.doc_id = doc_id
        self.description = description
        self.counter = counter

    def to_json(self):
        return self.__dict__

    def __str__(self):
        return json.dumps(self.__dict__)
