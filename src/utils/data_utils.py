"""
Shared data transformation utilities for preprocessing data.
These functions are used across data_transformation and model_evaluation components.
"""
import sys
import pandas as pd
from src.exception import MyException
from src.logger import logging


def map_gender_column(df: pd.DataFrame) -> pd.DataFrame:
    """Map Gender column to 0 for Female and 1 for Male."""
    try:
        logging.info("Mapping 'Gender' column to binary values")
        df = df.copy()
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
        return df
    except Exception as e:
        raise MyException(e, sys) from e


def create_dummy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create dummy variables for categorical features."""
    try:
        logging.info("Creating dummy variables for categorical features")
        return pd.get_dummies(df, drop_first=True)
    except Exception as e:
        raise MyException(e, sys) from e


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename specific columns and ensure integer types for dummy columns."""
    try:
        logging.info("Renaming specific columns and casting to int")
        df = df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype('int')
        return df
    except Exception as e:
        raise MyException(e, sys) from e


def drop_id_column(df: pd.DataFrame, column_name: str = "id") -> pd.DataFrame:
    """Drop the specified ID column if it exists."""
    try:
        logging.info(f"Dropping '{column_name}' column if present")
        if column_name in df.columns:
            df = df.drop(column_name, axis=1)
        # Also check for MongoDB's _id column
        if "_id" in df.columns:
            df = df.drop("_id", axis=1)
        return df
    except Exception as e:
        raise MyException(e, sys) from e
