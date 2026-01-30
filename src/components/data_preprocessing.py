import os
import sys
import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.exception import CustomException
from src.logger import logging
from src.utils.common import save_object

@dataclass
class DataPreprocessingConfig:
    """ Preprocessing configuration """
    preprocessor_path: str = "models/preprocessor.pkl"

class DataPreprocessing:
    """ Handles data cleaning and preprocessing """

    def __init__(self):
        self.config = DataPreprocessingConfig()
        logging.info("DataPreprocessing initialized")

    def get_preprocessor(self):
        """ Create preprocessing pipeline """
        
        try:
            logging.info("Creating preprocessor")

            # Numerical columns - apply scaling
            numerical_columns = ['Age', 'RegionID', 'AnnualPremium', 'SalesChannelID', 'DaysSinceCreated']

            # Categorical columns - apply encoding
            categorical_columns = ['Gender', 'HasDrivingLicense', 'Switch', 'VehicleAge', 'PastAccident']

            # Create preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers= [
                    ('num', StandardScaler(), numerical_columns),
                    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_columns)
                ],
                remainder='passthrough'
            )

            logging.info("Preprocessor created successfully")
            return preprocessor

        except Exception as e:
            logging.error("Error creating preprocessor")
            raise CustomException(e)
    
    def initiate_preprocessing(self, train_df, test_df):
        """ Apply preprocessing to train and test data """

        try:
            logging.info("Preprocessing started")

            # Drop ID cloumn
            train_df = train_df.drop(['id'], axis=1)
            test_df = test_df.drop(['id'], axis=1)

            # Fix AnnualPremium - remove commas and convert to float
            train_df['AnnualPremium'] = (
                train_df['AnnualPremium']
                .astype(str)
                .str.replace('£', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.strip()
            )

            train_df['AnnualPremium'] = pd.to_numeric(
                train_df['AnnualPremium'], errors='coerce'
            )

            test_df['AnnualPremium'] = (
                test_df['AnnualPremium']
                .astype(str)
                .str.replace('£', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.strip()
            )

            test_df['AnnualPremium'] = pd.to_numeric(
                test_df['AnnualPremium'], errors='coerce'
            )

            logging.info("Data cleaning started")

            # Handle missing values
            # NUmerical columns - fill with median
            num_cols = ['Age', 'RegionID', 'AnnualPremium', 'SalesChannelID', 
                       'DaysSinceCreated', 'HasDrivingLicense', 'Switch']
            for col in num_cols:
                if col in train_df.columns:
                    train_df[col] = train_df[col].fillna(train_df[col].median())
                    test_df[col] = test_df[col].fillna(test_df[col].median())
            
            # Categorical columns - fill with mode
            cat_cols = ['Gender', 'VehicleAge', 'PastAccident']
            for col in cat_cols:
                if col in train_df.columns:
                    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
                    test_df[col] = test_df[col].fillna(test_df[col].mode()[0])
            
            logging.info("Missing values handled")

            # Sepatate features and target
            target_column = 'Result'

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            logging.info(f"X_train Shape: {X_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}")

            # Get preprocessor
            preprocessor = self.get_preprocessor()

            # Fit on train, transform both
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info("Data transformation completed")

            # Save preprocessor
            save_object(self.config.preprocessor_path, preprocessor)
            logging.info(f"Preprocessor saved: {self.config.preprocessor_path}")

            return X_train_transformed, X_test_transformed, y_train, y_test
        
        except Exception as e:
            logging.error("Error in preprocessing")
            raise CustomException(e)

if __name__ == "__main__":
    # Test preprocessing
    from data_ingestion import DataIngestion

    # Get data
    ingestion = DataIngestion()
    train_df, test_df = ingestion.initiate_data_ingestion()

    # Preprocess
    preprocessing = DataPreprocessing()
    X_train, X_test, y_train, y_test = preprocessing.initiate_preprocessing(train_df, test_df)

    print(f"✅ X_train shape: {X_train.shape}")
    print(f"✅ X_test shape: {X_test.shape}")
    print(f"✅ y_train shape: {y_train.shape}")
    print(f"✅ y_test shape: {y_test.shape}")

            