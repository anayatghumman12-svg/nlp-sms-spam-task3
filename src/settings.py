from pathlib import Path

# Project Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "spam.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "spam_clean.csv"

# Encodings & Raw Column Mappings
RAW_CSV_ENCODING = "latin-1"
RAW_LABEL_COLUMN = "v1"
RAW_MESSAGE_COLUMN = "v2"

# Processed Column Names
LABEL_COLUMN = "label"
MESSAGE_COLUMN = "message"
CLEAN_MESSAGE_COLUMN = "clean_message"

# Label Mappings
LABEL_MAP = {"ham": 0, "spam": 1}
INV_LABEL_MAP = {0: "ham", 1: "spam"}
SPAM_LABEL = "spam"
HAM_LABEL = "ham"

# Model Hyperparameters
TEST_SIZE = 0.2          
RANDOM_STATE = 42        
TFIDF_MAX_FEATURES = 4000

# Output Paths
MODELS_DIR = PROJECT_ROOT / "models"
FINAL_PIPELINE_PATH = MODELS_DIR / "spam_pipeline.joblib"
RESULTS_DIR = PROJECT_ROOT / "results"