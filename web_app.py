import os
import uuid
from json import JSONEncoder

import httpagentparser  # for getting the user agent as json
from flask import Flask, render_template, session
from flask import request

from myapp.analytics.analytics_data import AnalyticsData
from myapp.search.load_corpus import load_corpus, load_preprocessed_corpus
from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SearchEngine
from myapp.generation.rag import RAGGenerator
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env


# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)
_default.default = JSONEncoder().default
JSONEncoder.default = _default
# end lines ***for using method to_json in objects ***


# instantiate the Flask application
app = Flask(__name__)

# random 'secret_key' is used for persisting data in secure cookie
app.secret_key = os.getenv("SECRET_KEY", "default-secret-key-change-me")
# open browser dev tool to see the cookies
app.session_cookie_name = os.getenv("SESSION_COOKIE_NAME", "irwa_session")
# instantiate our search engine
search_engine = SearchEngine()
# instantiate our in memory persistence
analytics_data = AnalyticsData()
# instantiate RAG generator
rag_generator = RAGGenerator()

# load documents corpus into memory.
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
file_path = path + "/" + os.getenv("DATA_FILE_PATH")
corpus = load_corpus(file_path)
preprocessed_corpus = load_preprocessed_corpus()
# Log first element of corpus to verify it loaded correctly:
print("\nCorpus is loaded... \n First element:\n", list(corpus.values())[0])


def get_session_id():
    """Get or create a session ID for analytics tracking"""
    if 'analytics_session_id' not in session:
        session['analytics_session_id'] = str(uuid.uuid4())
    return session['analytics_session_id']


# Home URL "/"
@app.route('/')
def index():
    print("starting home url /...")

    # Get or create session for analytics
    session_id = get_session_id()

    user_agent = request.headers.get('User-Agent')
    user_ip = request.remote_addr
    agent = httpagentparser.detect(user_agent)

    # Track session
    analytics_data.get_or_create_session(
        session_id=session_id,
        ip_address=user_ip,
        browser=agent.get('browser', {}).get('name'),
        os_name=agent.get('os', {}).get('name')
    )

    print("Remote IP: {} - Browser: {}".format(user_ip, agent))
    return render_template('index.html', page_title="Welcome")


@app.route('/search', methods=['POST'])
def search_form_post():
    search_query = request.form['search-query']
    algorithm = request.form.get('algorithm', 'tfidf')

    session['last_search_query'] = search_query
    session_id = get_session_id()

    # Use real search with selected algorithm
    results = search_engine.search(search_query, session_id, preprocessed_corpus, algorithm, top_k=20)

    found_count = len(results)

    # Track query with analytics
    search_id = analytics_data.save_query_terms(
        query_text=search_query,
        session_id=session_id,
        algorithm=algorithm,
        num_results=found_count
    )

    # Store search_id for click tracking
    session['last_search_id'] = search_id

    # generate RAG response based on user query and retrieved results
    rag_response = rag_generator.generate_response(search_query, results)
    print("RAG response:", rag_response[:100] if rag_response else "None")

    session['last_found_count'] = found_count

    return render_template('results.html', results_list=results, page_title="Results", found_counter=found_count, rag_response=rag_response, algorithm=algorithm)


@app.route('/doc_details', methods=['GET'])
def doc_details():
    """Show document details page with click tracking"""
    clicked_doc_id = request.args.get("pid")
    ranking_position = request.args.get("pos", type=int, default=0)

    if not clicked_doc_id:
        return render_template('doc_details.html', error="No document ID provided")

    print("click in id={}".format(clicked_doc_id))

    # Track click with analytics
    session_id = get_session_id()
    query_id = session.get('last_search_id')

    analytics_data.save_click(
        doc_id=clicked_doc_id,
        session_id=session_id,
        query_id=query_id,
        ranking_position=ranking_position
    )

    # Get document from preprocessed corpus
    if clicked_doc_id not in preprocessed_corpus:
        return render_template('doc_details.html', error=f"Document {clicked_doc_id} not found")

    doc_data = preprocessed_corpus[clicked_doc_id]
    original_data = doc_data.get("original", {})

    # Create Document object for rendering
    doc = Document(
        pid=clicked_doc_id,
        title=original_data.get("title", ""),
        description=original_data.get("description", ""),
        brand=original_data.get("brand"),
        category=original_data.get("category"),
        sub_category=original_data.get("sub_category"),
        product_details=original_data.get("product_details"),
        seller=original_data.get("seller"),
        out_of_stock=original_data.get("out_of_stock", False),
        selling_price=original_data.get("selling_price"),
        discount=original_data.get("discount"),
        actual_price=original_data.get("actual_price"),
        average_rating=original_data.get("average_rating"),
        url=original_data.get("url"),
        images=original_data.get("images")
    )

    return render_template('doc_details.html', doc=doc)


@app.route('/stats', methods=['GET'])
def stats():
    """Show simple statistics example with most clicked documents"""
    docs = []
    for doc_id in analytics_data.fact_clicks:
        if doc_id in preprocessed_corpus:
            row = preprocessed_corpus[doc_id]
            original_data = row.get("original", {})
            count = analytics_data.fact_clicks[doc_id]
            doc = StatsDocument(
                pid=doc_id,
                title=original_data.get("title", ""),
                description=original_data.get("description", ""),
                url=f"doc_details?pid={doc_id}",
                count=count
            )
            docs.append(doc)

    # Sort by click count descending
    docs.sort(key=lambda doc: doc.count, reverse=True)
    return render_template('stats.html', clicks_data=docs)


@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Analytics dashboard with metrics and visualizations"""
    # Get summary statistics
    stats = analytics_data.get_summary_stats()

    # Get top queries and clicked docs for display
    top_queries = analytics_data.get_top_queries(5)
    algo_dist = analytics_data.get_algorithm_distribution()

    return render_template('dashboard.html', stats=stats, top_queries=top_queries, algo_dist=algo_dist)

# Chart routes for dashboard
@app.route('/plot_number_of_views', methods=['GET'])
def plot_number_of_views():
    """Generate bar chart of document views"""
    return analytics_data.plot_number_of_views()


@app.route('/plot_top_queries', methods=['GET'])
def plot_top_queries():
    """Generate bar chart of top queries"""
    return analytics_data.plot_top_queries()


@app.route('/plot_algorithm_distribution', methods=['GET'])
def plot_algorithm_distribution():
    """Generate pie chart of algorithm usage"""
    return analytics_data.plot_algorithm_distribution()


if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=os.getenv("DEBUG"))
