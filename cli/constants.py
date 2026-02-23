import os

DEFAULT_SEARCH_LIMIT = 5

DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 0
DEFAULT_SEMANTIC_CHUNK_SIZE = 4
DEFAULT_CHUNK_SEARCH_LIMIT = 5

RETURN_DOCUMENT_LIMIT = 100

SCORE_PRECISION = 4


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'movies.json')
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, 'data', 'stopwords.txt')

CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache')

BM25_K1 = 1.5
BM25_B = 0.75