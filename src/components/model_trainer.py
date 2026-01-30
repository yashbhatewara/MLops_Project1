import sys
from typing import Tuple
from xml.parsers.expat import model

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.utils.main_utils import (get_classification_metrics,find_optimal_threshold)

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.entity.estimator import MyModel

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object,float]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function trains a RandomForestClassifier with specified parameters
        
        Output      :   Returns metric artifact object and trained model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Training RandomForestClassifier with specified parameters")

            # Splitting the train and test data into features and target variables
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            logging.info("train-test split done.")

            # Initialize RandomForestClassifier with specified parameters
            model = RandomForestClassifier(
                n_estimators = self.model_trainer_config._n_estimators,
                min_samples_split = self.model_trainer_config._min_samples_split,
                min_samples_leaf = self.model_trainer_config._min_samples_leaf,
                max_depth = self.model_trainer_config._max_depth,
                criterion = self.model_trainer_config._criterion,
                class_weight="balanced",  # 🔥 CRITICAL FIX
                random_state=self.model_trainer_config._random_state,
                n_jobs=-1
            )
            
            # Fit the model
            logging.info("Model training going on...")
            model.fit(x_train, y_train)
            logging.info("Model training done.")

            # Predictions and evaluation metrics
            y_train_proba = model.predict_proba(x_train)[:, 1]
            y_test_proba = model.predict_proba(x_test)[:, 1]

            # Find optimal threshold on training data
            optimal_threshold = find_optimal_threshold(
                y_true=y_train,
                y_pred_proba=y_train_proba,
                metric="f1"
            )

            logging.info(f"Optimal threshold selected: {optimal_threshold:.4f}")

            # Apply threshold on test data
            y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

            # Compute metrics
            metrics = get_classification_metrics(
                y_true=y_test,
                y_pred=y_test_pred,
                y_pred_proba=y_test_proba
            )

            metric_artifact = ClassificationMetricArtifact(
                f1_score=metrics["f1_score"],
                precision_score=metrics["precision"],
                recall_score=metrics["recall"]
            )

            # ✅ RETURN AT THE VERY END
            return model, metric_artifact, optimal_threshold

      
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model training steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            print("------------------------------------------------------------------------------------------------")
            print("Starting Model Trainer Component")
            # Load transformed train and test data
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("train-test data loaded")
            
            # Train model and get metrics
            trained_model, metric_artifact, optimal_threshold = self.get_model_object_and_report(train=train_arr, test=test_arr)
            logging.info("Model object and artifact loaded.")
            
            # Load preprocessing object
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing obj loaded.")

            # Save the final model object that includes both preprocessing and the trained model
            logging.info("Saving new model as performace is better than previous one.")
            model_package = {"preprocessing_object": preprocessing_obj,"trained_model": trained_model,
                             "threshold": optimal_threshold} 

            save_object(self.model_trainer_config.trained_model_file_path, model_package)

            save_object(self.model_trainer_config.trained_model_file_path, model_package)
            logging.info("Saved final model object that includes both preprocessing and the trained model")

            # Create and return the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                 trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                 metric_artifact=metric_artifact,
                 threshold=optimal_threshold
            )

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e