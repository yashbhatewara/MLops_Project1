import sys
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from src.utils.main_utils import (get_classification_metrics,find_optimal_threshold)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.entity.estimator import MyModel
from src.constants import *

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
        Description :   This function trains RandomForestClassifier or LightGBM with specified parameters
        
        Output      :   Returns metric artifact object and trained model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Training RandomForestClassifier with specified parameters")

            # Splitting the train and test data into features and target variables
            
            x_full, y_full = train[:, :-1], train[:, -1]

            x_train, x_calib, y_train, y_calib =train_test_split(x_full,y_full,
                test_size=0.2,random_state=42,stratify=y_full)

            x_test, y_test = test[:, :-1], test[:, -1]
            logging.info("train-test split done.")

            # Initialize RandomForestClassifier with specified parameters
            if MODEL_TYPE == "random_forest":
                logging.info("Training RandomForest model")
                model = RandomForestClassifier(
                    n_estimators=MODEL_TRAINER_N_ESTIMATORS,
                    min_samples_split=MODEL_TRAINER_MIN_SAMPLES_SPLIT,
                    min_samples_leaf=MODEL_TRAINER_MIN_SAMPLES_LEAF,
                    max_depth=MIN_SAMPLES_SPLIT_MAX_DEPTH,
                    criterion=MIN_SAMPLES_SPLIT_CRITERION,
                    class_weight="balanced",
                    random_state=MIN_SAMPLES_SPLIT_RANDOM_STATE,
                    n_jobs=-1
                )

            elif MODEL_TYPE == "LightGBM":
                logging.info("Training LightGBM model")
                model = lgb.LGBMClassifier(
                    n_estimators=LGBM_N_ESTIMATORS,
                    learning_rate=LGBM_LEARNING_RATE,
                    max_depth=LGBM_MAX_DEPTH,
                    num_leaves=LGBM_NUM_LEAVES,
                    subsample=LGBM_SUBSAMPLE,
                    colsample_bytree=LGBM_COLSAMPLE_BYTREE,
                    class_weight=LGBM_CLASS_WEIGHT,
                    random_state=LGBM_RANDOM_STATE,
                    n_jobs=-1
                )
            else:
                raise MyException(
                    f"Unsupported MODEL_TYPE: {MODEL_TYPE}. Expected 'random_forest' or 'LightGBM'",
                sys
                )
            
            # Fit the model
            logging.info("Model training going on...")
            model.fit(x_train, y_train)
            logging.info("Model training done.")

            logging.info("Applying probability calibration (sigmoid)")
            calibrated_model = CalibratedClassifierCV(
                estimator=model,
                method="sigmoid",
                cv="prefit"
            )
            calibrated_model.fit(x_calib, y_calib)

            # Predictions and evaluation metrics
            # Using calibration set for threshold selection
            y_calib_proba = calibrated_model.predict_proba(x_calib)[:, 1]
            y_test_proba = calibrated_model.predict_proba(x_test)[:, 1]

            # Find optimal threshold on training data
            optimal_threshold = find_optimal_threshold(
                y_true=y_calib,
                y_pred_proba=y_calib_proba,
                metric=THRESHOLD_OPTIMIZATION_METRIC
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

            # ✅ RETURN AT THE VERY END (return calibrated model, not base model)
            return calibrated_model, metric_artifact, optimal_threshold

      
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
            calibrated_model, metric_artifact, optimal_threshold = self.get_model_object_and_report(train=train_arr, test=test_arr)
            logging.info("Model object and artifact loaded.")
            
            # Load preprocessing object
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing obj loaded.")

            # Save the final model object that includes both preprocessing and the trained model
            logging.info("Saving new model as performace is better than previous one.")
            model_package = {"preprocessing_object": preprocessing_obj,"trained_model": calibrated_model,
                             "threshold": optimal_threshold} 

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