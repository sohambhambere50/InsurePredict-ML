import os 
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class TrainingPipeline:
    """ Complete training pipeline """

    def __init__(self):
        logging.info("Training Pipeline initialized")
    
    def run_pipeline(self):
        """ Execute complete training pipeline """

        try:
            # Step 1: Data Ingestion
            logging.info("=" * 50)
            logging.info("Step 1: Data Ingestion")
            ingestion = DataIngestion()
            train_df, test_df = ingestion.initiate_data_ingestion()

            # Step 2: Data Preprocessing
            logging.info("=" * 50)
            logging.info("Step 2: Data Preprocessing")
            preprocessing = DataPreprocessing()
            X_train, X_test, y_train, y_test = preprocessing.initiate_preprocessing(train_df, test_df)

            # Step 3: Model Training
            logging.info("=" * 50)
            logging.info("Step 3: Model Training")
            trainer = ModelTrainer()
            metrics = trainer.train_model(X_train, y_train, X_test, y_test)

            logging.info("=" * 50)
            logging.info("Tracking Pipeline completed successfully")
            logging.info("=" * 50)

            return metrics
        except Exception as e:
            logging.info("Error in training pipeline")
            raise CustomException(e)
        
if __name__ == "__main__":
    pipeline = TrainingPipeline()
    metrics = pipeline.run_pipeline()

    print("\n" + "=" * 50)
    print("TRAINING COMPLETED!")
    print("=" * 50)
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test Precision: {metrics['test_precision']:.4f}")
    print(f"Test Recall: {metrics['test_recall']:.4f}")
    print(f"Test F1 Score: {metrics['test_f1']:.4f}")
    print(f"Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
    print("=" * 50)