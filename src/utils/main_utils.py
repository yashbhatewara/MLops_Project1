import os
import sys

import numpy as np
import dill
import yaml
from pandas import DataFrame
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve
)

from src.exception import MyException
from src.logger import logging


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise MyException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise MyException(e, sys) from e


def load_object(file_path: str) -> object:
    """
    Returns model/object from project directory.
    file_path: str location of file to load
    return: Model/Obj
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        return obj
    except Exception as e:
        raise MyException(e, sys) from e

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise MyException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise MyException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method of utils")

    except Exception as e:
        raise MyException(e, sys) from e
    

def apply_smote_balancing(X_train: np.ndarray,
                          y_train: np.ndarray,
                          sampling_strategy: float = 0.30,
                          k_neighbors: int = 5,
                          random_state: int = 42 ):
    """
    Apply SMOTE only when class imbalance is significant
    """
    try:
        unique, counts = np.unique(y_train, return_counts=True)
        class_ratio = min(counts) / sum(counts)

        logging.info(f"Class distribution before SMOTE: {dict(zip(unique, counts))}")
        logging.info(f"Minority class ratio: {class_ratio:.2%}")

        if class_ratio < 0.30:
            logging.info("Applying SMOTE for class balancing")

            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=42
            )

            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

            unique, counts = np.unique(y_resampled, return_counts=True)
            logging.info(f"Class distribution after SMOTE: {dict(zip(unique, counts))}")

            return X_resampled, y_resampled

        logging.info("SMOTE not required")
        return X_train, y_train

    except Exception as e:
        raise MyException(e, sys) from e


def get_classification_metrics(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_pred_proba: np.ndarray = None) -> dict:
    """
    Calculate classification metrics used across pipeline
    """
    try:
        metrics = {
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred)
        }

        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = cm.ravel()

        if y_pred_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)

        logging.info(
            f"Metrics | Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, "
            f"F1: {metrics['f1_score']:.4f}"
        )

        return metrics

    except Exception as e:
        raise MyException(e, sys) from e


def find_optimal_threshold(y_true: np.ndarray,
                           y_pred_proba: np.ndarray,
                           metric: str = "f1") -> float:
    """
    Find optimal decision threshold based on precision-recall curve.
    
    Supported metrics: 'f1', 'f0.5', 'f0.75', 'f2', 'precision', 'recall'
    """
    try:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

        # Calculate scores based on the specified metric
        metric_lower = metric.lower()
        
        if metric_lower == "f1":
            scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        elif metric_lower == "f0.5":
            # F0.5 weighs precision higher than recall
            beta = 0.5
            scores = (1 + beta**2) * (precisions * recalls) / ((beta**2 * precisions) + recalls + 1e-10)
        elif metric_lower == "f0.75":
            # F0.75 weighs precision slightly higher than recall
            beta = 0.75
            scores = (1 + beta**2) * (precisions * recalls) / ((beta**2 * precisions) + recalls + 1e-10)
        elif metric_lower == "f2":
            # F2 weighs recall higher than precision
            beta = 2
            scores = (1 + beta**2) * (precisions * recalls) / ((beta**2 * precisions) + recalls + 1e-10)
        elif metric_lower == "precision":
            scores = precisions
        elif metric_lower == "recall":
            scores = recalls
        else:
            logging.warning(f"Unknown metric '{metric}', defaulting to F1")
            scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        
        best_idx = np.argmax(scores)

        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        logging.info(
            f"Optimal threshold found: {optimal_threshold:.4f} "
            f"(metric={metric}, score={scores[best_idx]:.4f})"
        )

        return optimal_threshold

    except Exception as e:
        raise MyException(e, sys) from e



# def drop_columns(df: DataFrame, cols: list)-> DataFrame:

#     """
#     drop the columns form a pandas DataFrame
#     df: pandas DataFrame
#     cols: list of columns to be dropped
#     """
#     logging.info("Entered drop_columns methon of utils")

#     try:
#         df = df.drop(columns=cols, axis=1)

#         logging.info("Exited the drop_columns method of utils")
        
#         return df
#     except Exception as e:
#         raise MyException(e, sys) from e