import os

RAW_DIR = "artifacts/raw"
RAW_FILE = os.path.join(RAW_DIR, "raw.csv")
TRAIN_FILE = os.path.join(RAW_DIR, "train.csv")
TEST_FILE = os.path.join(RAW_DIR, "test.csv")

CONFIG_PATH = "config/config.yaml"


#### Data Processing ####

PROCESSED_DIR = "artifacts/processed"
PROCESSED_TRAIN_FILE = os.path.join(PROCESSED_DIR, "train_processed.csv")
PROCESSED_TEST_FILE = os.path.join(PROCESSED_DIR, "test_processed.csv")

#### Model Training ####
MODEL_OUTPUT_PATH = "artifacts/models/lgbm_model.pkl"