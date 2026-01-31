import os 
import sys
import mlflow
from mlflow.sklearn import log_model
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.exception import CustomException
from src.logger import logging
from src.utils.common import save_object
from src.utils.common import read_yaml

@dataclass
class ModelTrainerConfig:
    """ Model training configuration """
    model_path: str = 'models/model.pkl'

class ModelTrainer:
    """ Handles model training and evaluation """

    def __init__(self):
        self.config = ModelTrainerConfig()
        # Load config from YAML
        self.params = read_yaml('config.yaml')
        logging.info("ModelTrainer initialized")

    def train_model(self, X_train, y_train, X_test, y_test):
        """ Train and evaluate model with MLflow tracking """

        try:
            logging.info("Model training started")

            with mlflow.start_run():            
                # Initialize model with config parameters
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                )
                
                # Log parameters from config
                mlflow.log_param("model_type", "RandomForestClassifier")
                mlflow.log_param("n_estimators", 100)
                mlflow.log_param("max_depth", 10)
                mlflow.log_param("class_weight", "balanced")

                # Train model
                logging.info("Fitting model on training data")
                model.fit(X_train, y_train)

                # Predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Calculate metrics
                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)            
                test_precision = precision_score(y_test, y_test_pred)
                test_recall = recall_score(y_test, y_test_pred)
                test_f1 = f1_score(y_test, y_test_pred)
                test_roc_auc = roc_auc_score(y_test, y_test_pred)
                
                # Log metrics
                mlflow.log_metric("train_accuracy", float(train_accuracy))
                mlflow.log_metric("test_accuracy", float(test_accuracy))
                mlflow.log_metric("test_precision", float(test_precision))
                mlflow.log_metric("test_recall", float(test_recall))
                mlflow.log_metric("test_f1", float(test_f1))
                mlflow.log_metric("test_roc_auc", float(test_roc_auc))

                # Log model
                log_model(model, "model")
                    
                # Log to console
                logging.info(f"Train Accuracy: {train_accuracy:.4f}")
                logging.info(f"Test Accuracy: {test_accuracy:.4f}")
                logging.info(f"Test Precision: {test_precision:.4f}")
                logging.info(f"Test Recall: {test_recall:.4f}")
                logging.info(f"Test F1 Score: {test_f1:.4f}")
                logging.info(f"Test ROC-AUC: {test_roc_auc:.4f}")
                
                # Save model
                save_object(self.config.model_path, model)
                logging.info(f"Model saved: {self.config.model_path}")

                return {
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'test_precision': test_precision,
                    'test_recall': test_recall,
                    'test_f1': test_f1,
                    'test_roc_auc': test_roc_auc
                }
        
        except Exception as e:
            logging.error("Error in model training")
            raise CustomException(e)

if __name__ == "__main__":
    # Test model training
    from data_ingestion import DataIngestion
    from data_preprocessing import DataPreprocessing

    # Pipeling
    print("Step 1: Data Ingestion")
    ingestion = DataIngestion()
    train_df, test_df = ingestion.initiate_data_ingestion()

    print("Step 2: Data Preprocessing")
    preprocessing = DataPreprocessing()
    X_train, X_test, y_train, y_test = preprocessing.initiate_preprocessing(train_df, test_df)

    print("Step 3: Model Training")
    trainer = ModelTrainer()
    metrics = trainer.train_model(X_train, y_train, X_test, y_test)

    print("\n✅ Model Training Complete!")
    print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test F1 Score: {metrics['test_f1']:.4f}")





