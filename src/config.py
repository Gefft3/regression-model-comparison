import os

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
RAW_DATA_PATH = os.getenv('RAW_DATA_PATH', os.path.join(BASE_DIR, 'data', 'raw'))
PROCESSED_DATA_PATH = os.getenv('PROCESSED_DATA_PATH', os.path.join(BASE_DIR, 'data', 'processed'))
MODEL_DIR = os.getenv('MODEL_DIR', os.path.join(BASE_DIR, 'models'))
MODEL_FILENAME = os.getenv('MODEL_FILENAME', 'best_pipeline.joblib')
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)