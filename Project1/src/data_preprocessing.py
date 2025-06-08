import os, sys
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataPreprocessor:
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config_path = config_path
        
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            logger.info(f"Created directory: {self.processed_dir}")

    def preprocess_data(self, df):
        try:
            logger.info("Preprocessing data...")
            logger.info("Dropping the columns")
            df.drop(columns=['Unnamed: 0', 'Booking_ID'], inplace=True)
            df.drop_duplicates(inplace=True)
            
            cat_cols= self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]
            
            logger.info("Encoding categorical columns")
            
            label_encoders = LabelEncoder()
            mappings = {}
            
            for col in cat_cols:
                df[col] = label_encoders.fit_transform(df[col])
                mappings[col] = dict(zip(label_encoders.classes_, label_encoders.transform(label_encoders.classes_)))
                
            logger.info("Label encoding completed")
            for col, mapping in mappings.items():
                logger.info(f"Mapping for {col}: {mapping}")
                
            logger.info("Doing Skewness handling")
            
            skew_threshold = self.config["data_processing"]["skewness_threshold"]
            skewness = df[num_cols].apply(lambda x: x.skew())
            
            for col in skewness[skewness.abs() > skew_threshold].index:
                df[col] = np.log1p(df[col])
                logger.info(f"Applied log transformation to {col} due to high skewness")    
            
            
            return df
        
        except Exception as e:  
            logger.error(f"Error during data preprocessing: {e}")
            raise CustomException(e, sys)
        
    def balance_data(self, df):
        try:
            logger.info("Balancing data using SMOTE...")
            X = df.drop(columns=['booking_status'])
            y = df['booking_status']

            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df['booking_status'] = y_resampled

            logger.info("Data balancing completed")
            return balanced_df

        except Exception as e:
            logger.error(f"Error during data balancing: {e}")
            raise CustomException(e, sys)
        
    def select_features(self, df):
        try:
            logger.info("Selecting features using Random Forest Classifier...")
            X = df.drop(columns=['booking_status'])
            y = df['booking_status']

            rf = RandomForestClassifier(random_state=42)
            rf.fit(X, y)

            feature_importance = rf.feature_importances_
            future_importance_df = pd.DataFrame({
                'feature': X.columns,
                'Importance': feature_importance
            })
            
            top_features_importance_df = future_importance_df.sort_values(by='Importance', ascending=False)
            
            num_top_features = self.config["data_processing"]["num_top_features"]
            
            top_features = top_features_importance_df['feature'].head(num_top_features).values
            
            logger.info(f"Selected top {num_top_features} features: {top_features}")
               
            top_10_features_df = df[top_features.tolist() + ['booking_status']]
            logger.info("Feature selection completed")
            return top_10_features_df
        except Exception as e:
            logger.error(f"Error during feature selection: {e}")
            raise CustomException(e, sys)
        
        
    def save_processed_data(self, df, file_path):
        try:
            file_path = os.path.join(self.processed_dir, file_path)
            df.to_csv(file_path, index=False)
            logger.info(f"Processed data saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise CustomException(e, sys)
        
    def process(self):
        try:
            logger.info("Starting data preprocessing pipeline...")

            self.config = read_yaml(self.config_path)

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            logger.info("Preprocessing training data")
            train_df = self.preprocess_data(train_df)
            train_df = self.balance_data(train_df)
            train_df = self.select_features(train_df)
            self.save_processed_data(train_df, 'train_processed.csv')

            logger.info("Preprocessing test data")
            test_df = self.preprocess_data(test_df)
            test_df = self.balance_data(test_df)
            test_df = test_df[train_df.columns] #alternative way of feature selection
            #test_df = self.select_features(test_df)
            self.save_processed_data(test_df, 'test_processed.csv')

            logger.info("Data preprocessing pipeline completed successfully")

        except Exception as e:
            logger.error(f"Error in data preprocessing pipeline: {e}")
            raise CustomException(e, sys)
        
        finally:
            logger.info("Data preprocessing pipeline finished")
            
            
    
    
if __name__ == "__main__":
    processor = DataPreprocessor(
        train_path=TRAIN_FILE,
        test_path=TEST_FILE,
        processed_dir=PROCESSED_DIR,
        config_path=CONFIG_PATH
    )
    processor.process()
        
              