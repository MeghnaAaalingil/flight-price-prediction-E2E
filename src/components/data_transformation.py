import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, threshold=1.5):
        self.columns = columns
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for column in self.columns:
            q1 = X_copy[column].quantile(0.25)
            q3 = X_copy[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - self.threshold * iqr
            upper_bound = q3 + self.threshold * iqr
            X_copy = X_copy[(X_copy[column] >= lower_bound) & (X_copy[column] <= upper_bound)]
        return X_copy

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            categorical_columns = [
                "airline",
                "source_city",
                "departure_time",
                "stops",
                "arrival_time",
                "destination_city",
                "class"
            ]
            
            ordinal_columns = ["departure_time", "arrival_time"]
            ordinal_encoder = OrdinalEncoder()

            # One-Hot Encoding for all other categorical variables
            cat_columns_to_encode = list(set(categorical_columns) - set(ordinal_columns))
            one_hot_pipeline = Pipeline(
                steps=[("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', drop='first'))]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("ordinal", ordinal_encoder, ordinal_columns),
                    ("one_hot", one_hot_pipeline, cat_columns_to_encode)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Remove unwanted columns
            columns_to_remove = ["flight"]
            train_df.drop(columns=columns_to_remove, errors='ignore', inplace=True)
            test_df.drop(columns=columns_to_remove, errors='ignore', inplace=True)

            logging.info("Unwanted columns removed")

            # Remove outliers
            outlier_columns = ["price"]
            outlier_remover = OutlierRemover(columns=outlier_columns)
            train_df = outlier_remover.transform(train_df)
            test_df = outlier_remover.transform(test_df)

            logging.info("Outliers removed")

            # Obtain preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "price"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr.toarray(), np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr.toarray(), np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
