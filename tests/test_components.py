import pytest
import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer


class TestDataIngestion:
    """Test data ingestion component"""
    
    def test_data_ingestion_creates_dataframes(self):
        """Test if data ingestion returns valid dataframes"""
        ingestion = DataIngestion()
        train_df, test_df = ingestion.initiate_data_ingestion()
        
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        assert len(train_df) > 0
        assert len(test_df) > 0
    
    def test_data_has_required_columns(self):
        """Test if data has all required columns"""
        ingestion = DataIngestion()
        train_df, test_df = ingestion.initiate_data_ingestion()
        
        required_columns = ['Gender', 'Age', 'Result']
        
        for col in required_columns:
            assert col in train_df.columns
            assert col in test_df.columns


class TestDataPreprocessing:
    """Test data preprocessing component"""
    
    def test_preprocessing_shapes(self):
        """Test if preprocessing returns correct shapes"""
        # Get data
        ingestion = DataIngestion()
        train_df, test_df = ingestion.initiate_data_ingestion()
        
        # Preprocess
        preprocessing = DataPreprocessing()
        X_train, X_test, y_train, y_test = preprocessing.initiate_preprocessing(train_df, test_df)
        
        # Check shapes
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        assert X_train.shape[1] == X_test.shape[1]
    
    def test_no_missing_values_after_preprocessing(self):
        """Test if missing values are handled"""
        ingestion = DataIngestion()
        train_df, test_df = ingestion.initiate_data_ingestion()
        
        preprocessing = DataPreprocessing()
        X_train, X_test, y_train, y_test = preprocessing.initiate_preprocessing(train_df, test_df)
        
        # Check for NaN values
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()
    
    def test_preprocessor_saved(self):
        """Test if preprocessor is saved"""
        ingestion = DataIngestion()
        train_df, test_df = ingestion.initiate_data_ingestion()
        
        preprocessing = DataPreprocessing()
        preprocessing.initiate_preprocessing(train_df, test_df)
        
        assert os.path.exists("models/preprocessor.pkl")


class TestModelTrainer:
    """Test model training component"""
    
    def test_model_training_returns_metrics(self):
        """Test if model training returns metrics dictionary"""
        # Get and preprocess data
        ingestion = DataIngestion()
        train_df, test_df = ingestion.initiate_data_ingestion()
        
        preprocessing = DataPreprocessing()
        X_train, X_test, y_train, y_test = preprocessing.initiate_preprocessing(train_df, test_df)
        
        # Train model
        trainer = ModelTrainer()
        metrics = trainer.train_model(X_train, y_train, X_test, y_test)
        
        # Check metrics exist
        assert 'test_accuracy' in metrics
        assert 'test_f1' in metrics
        assert 'test_roc_auc' in metrics
    
    def test_model_accuracy_threshold(self):
        """Test if model meets minimum accuracy threshold"""
        ingestion = DataIngestion()
        train_df, test_df = ingestion.initiate_data_ingestion()
        
        preprocessing = DataPreprocessing()
        X_train, X_test, y_train, y_test = preprocessing.initiate_preprocessing(train_df, test_df)
        
        trainer = ModelTrainer()
        metrics = trainer.train_model(X_train, y_train, X_test, y_test)
        
        # Model should have at least 60% accuracy
        assert metrics['test_accuracy'] > 0.60
    
    def test_model_saved(self):
        """Test if model is saved"""
        ingestion = DataIngestion()
        train_df, test_df = ingestion.initiate_data_ingestion()
        
        preprocessing = DataPreprocessing()
        X_train, X_test, y_train, y_test = preprocessing.initiate_preprocessing(train_df, test_df)
        
        trainer = ModelTrainer()
        trainer.train_model(X_train, y_train, X_test, y_test)
        
        assert os.path.exists("models/model.pkl")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])