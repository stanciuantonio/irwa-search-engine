"""
Analytics Data Module - Web Analytics for IRWA Search Engine

TODO: Future improvements to implement:
- [ ] Full star schema with Session, Query, Click, Request tables
- [ ] Pickle persistence for data between restarts
- [ ] Dwell time calculation (time between click and return)
- [ ] User agent/browser tracking
- [ ] CTR by ranking position
- [ ] Hourly activity tracking
"""

import json
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

import altair as alt
import pandas as pd


class AnalyticsData:
    """
    In-memory analytics storage.
    Tracks basic metrics: clicks, queries, sessions.
    """

    def __init__(self):
        # Click counter per document
        self.fact_clicks: Dict[str, int] = {}

        # Query tracking
        self.queries: List[Dict] = []

        # Session tracking (basic)
        self.sessions: Dict[str, Dict] = {}

        # Click details for analytics
        self.click_details: List[Dict] = []

        self._query_counter = 0

    def save_query_terms(self, query_text: str, session_id: str = None,
                         algorithm: str = 'tfidf', num_results: int = 0) -> int:
        """Save a search query and return query_id"""
        self._query_counter += 1

        self.queries.append({
            'query_id': self._query_counter,
            'query_text': query_text,
            'session_id': session_id,
            'algorithm': algorithm,
            'num_results': num_results,
            'timestamp': datetime.now().isoformat()
        })

        return self._query_counter

    def save_click(self, doc_id: str, session_id: str = None,
                   query_id: int = None, ranking_position: int = None):
        """Save a document click"""
        # Update click counter
        if doc_id in self.fact_clicks:
            self.fact_clicks[doc_id] += 1
        else:
            self.fact_clicks[doc_id] = 1

        # Save click details
        self.click_details.append({
            'doc_id': doc_id,
            'session_id': session_id,
            'query_id': query_id,
            'ranking_position': ranking_position,
            'timestamp': datetime.now().isoformat()
        })

    def get_or_create_session(self, session_id: str, ip_address: str = None,
                               browser: str = None, os_name: str = None) -> Dict:
        """Get or create a session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'session_id': session_id,
                'ip_address': ip_address,
                'browser': browser,
                'os': os_name,
                'start_time': datetime.now().isoformat()
            }
        return self.sessions[session_id]

    # Summary statistics
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for dashboard"""
        return {
            'total_sessions': len(self.sessions),
            'total_queries': len(self.queries),
            'total_clicks': len(self.click_details),
            'unique_docs_clicked': len(self.fact_clicks)
        }

    def get_top_queries(self, limit: int = 10) -> List[Dict]:
        """Get most frequent queries"""
        query_counts = defaultdict(int)
        for q in self.queries:
            query_counts[q['query_text']] += 1
        sorted_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)
        return [{'query': q, 'count': c} for q, c in sorted_queries[:limit]]

    def get_algorithm_distribution(self) -> Dict[str, int]:
        """Get distribution of algorithms used"""
        algo_counts = defaultdict(int)
        for q in self.queries:
            algo_counts[q.get('algorithm', 'unknown')] += 1
        return dict(algo_counts)

    # Visualization Methods
    def plot_number_of_views(self):
        """Bar chart of document views"""
        data = [{'Document ID': doc_id[:12] + '...' if len(doc_id) > 12 else doc_id, 'Views': count}
                for doc_id, count in sorted(self.fact_clicks.items(),
                                           key=lambda x: x[1], reverse=True)[:10]]
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
