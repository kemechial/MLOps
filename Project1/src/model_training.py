import os, sys
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml, load_data
from scipy.stats import randint

import mlflow
import mlflow.sklearn


logger = get_logger(__name__)

class ModelTraining:
    
    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        
        self.params_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS
        
    def load_and_split_data(self):
        try:
            logger.info(f"Loading data from {self.train_path} and {self.test_path}")
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            X_train = train_df.drop(columns=['booking_status'])
            y_train = train_df['booking_status']
            X_test = test_df.drop(columns=['booking_status'])
            y_test = test_df['booking_status']
            logger.info("Data loaded and split successfully.")
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error(f"Error loading or splitting data: {e}")     
            raise CustomException(e, sys)
        
    def train_lgbm(self, X_train, y_train):
        try:
            logger.info("Starting LightGBM model training.")
            lgbm_model = lgb.LGBMClassifier(random_state=self.random_search_params["random_state"])
            logger.info("Starting our Hyperparameter tuning")
            
            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params["n_iter"],
                cv=self.random_search_params["cv"],
                n_jobs=self.random_search_params["n_jobs"],
                verbose=self.random_search_params["verbose"],
                random_state=self.random_search_params["random_state"],
                scoring=self.random_search_params["scoring"]
            )
             
            logger.info("Starting our model training")
            
            random_search.fit(X_train, y_train)
            
            logger.info("Model training completed successfully.")
            
            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_
            
            logger.info(f"Best parameters found: {best_params}")
            
            return best_lgbm_model
        
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise CustomException(e, sys)
    
    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating model performance...")
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            logger.info(f"Model evaluation completed. Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
            
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise CustomException(e, sys)
        
    def save_model(self, model):
        try:
            logger.info("Saving trained model...")
            os.makedirs(os.path.dirname(self.model_output_path) , exist_ok=True)
            
            logger.info(f"Saving model to {self.model_output_path}")
            joblib.dump(model, self.model_output_path)
            logger.info(f"Model saved successfully at {self.model_output_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise CustomException(e, sys)
        
    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting model training pipeline")
                logger.info("Starting our MLFLOW experiment")
                
                logger.info("Logging the training and testing dataset to MLFLOW")
                mlflow.log_artifact(self.train_path, artifact_path="datasets")
                mlflow.log_artifact(self.test_path, artifact_path="datasets" )
                
                X_train, y_train, X_test, y_test = self.load_and_split_data()
                best_lgbm_model = self.train_lgbm(X_train, y_train)
                metrics = self.evaluate_model(best_lgbm_model, X_test, y_test)
                self.save_model(best_lgbm_model)
                
                logger.info("Logging the model into MLFLOW")
                mlflow.log_artifact(self.model_output_path)
                
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(metrics)
                
            
                logger.info("Model training pipeline completed successfully.")
            
        except Exception as e:
            logger.error(f"Error in model training pipeline: {e}")
            raise CustomException(e, sys)
        
        finally:
            logger.info("Model training process finished.") 
            
            
if __name__ == "__main__":
    trainer = ModelTraining(PROCESSED_TRAIN_FILE, PROCESSED_TEST_FILE, MODEL_OUTPUT_PATH)
    trainer.run()
    