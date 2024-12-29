import os

BASE_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
OUTPUT_PATH = BASE_PATH + '/output'
EMBEDDING_PATH = OUTPUT_PATH + '/pairs'
DATA_PATH = BASE_PATH + '/data/'
MODELS_PATH = BASE_PATH + '/models/'
LOG_PATH = BASE_PATH + '/logs/'
CHECKPOINT_PATH = OUTPUT_PATH + '/checkpoints'