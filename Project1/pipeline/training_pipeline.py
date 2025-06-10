from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTraining

from utils.common_functions import read_yaml
from config.paths_config import *


if __name__ == '__main__':
    
    ### Data Ingestion ###
    
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()
    
    
    ### Data Processing ###
    
    processor = DataPreprocessor(
        train_path=TRAIN_FILE,
        test_path=TEST_FILE,
        processed_dir=PROCESSED_DIR,
        config_path=CONFIG_PATH
    )
    processor.process()
    
    ### Model Training ###
    
    trainer = ModelTraining(PROCESSED_TRAIN_FILE, PROCESSED_TEST_FILE, MODEL_OUTPUT_PATH)
    trainer.run()
    
    
           
    