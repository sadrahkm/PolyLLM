import os

BASE_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
EMBEDDING_PATH = BASE_PATH + '/output/pairs'
DATA_PATH = BASE_PATH + '/data/'
MODELS_PATH = BASE_PATH + '/models/'
LOG_PATH = BASE_PATH + '/logs/'