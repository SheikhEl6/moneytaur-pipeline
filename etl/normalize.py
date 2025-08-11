#!/usr/bin/env python3
"""
Data Normalization Module for MoneyTaur Pipeline

This module handles the ETL (Extract, Transform, Load) processes for financial data,
focusing on data normalization, cleaning, and standardization.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataNormalizer:
    """
    Handles data normalization and standardization for financial datasets.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the DataNormalizer.
        
        Args:
            config: Configuration dictionary for normalization parameters
        """
        self.config = config or {
            'date_format': '%Y-%m-%d',
            'numeric_precision': 6,
            'missing_value_threshold': 0.1
        }
        logger.info("DataNormalizer initialized")
    
    def load_raw_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load raw data from file.
        
        Args:
            file_path: Path to the raw data file
            
        Returns:
            DataFrame containing raw data
        """
        logger.info(f"Loading raw data from {file_path}")
        try:
            if str(file_path).endswith('.csv'):
                return pd.read_csv(file_path)
            elif str(file_path).endswith('.json'):
                return pd.read_json(file_path)
            else:
                # TODO: Add support for other file formats
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names and types.
        
        Args:
            df: DataFrame to standardize
            
        Returns:
            DataFrame with standardized columns
        """
        logger.info("Standardizing column names and types")
        
        # Copy dataframe to avoid modifying original
        df_normalized = df.copy()
        
        # Standardize column names (lowercase, replace spaces with underscores)
        df_normalized.columns = (
            df_normalized.columns
            .str.lower()
            .str.replace(' ', '_')
            .str.replace('-', '_')
        )
        
        # TODO: Implement column type standardization logic
        return df_normalized
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values")
        
        df_clean = df.copy()
        
        # Check missing value percentage for each column
        missing_pct = df_clean.isnull().sum() / len(df_clean)
        threshold = self.config['missing_value_threshold']
        
        # Drop columns with too many missing values
        cols_to_drop = missing_pct[missing_pct > threshold].index
        if not cols_to_drop.empty:
            logger.warning(f"Dropping columns with >{threshold*100}% missing values: {cols_to_drop.tolist()}")
            df_clean = df_clean.drop(columns=cols_to_drop)
        
        # TODO: Implement more sophisticated missing value handling
        # For now, forward fill numeric columns and drop remaining nulls
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method='ffill')
        
        return df_clean.dropna()
    
    def normalize_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numeric columns (scaling, rounding, etc.).
        
        Args:
            df: DataFrame with numeric columns to normalize
            
        Returns:
            DataFrame with normalized numeric columns
        """
        logger.info("Normalizing numeric columns")
        
        df_norm = df.copy()
        numeric_cols = df_norm.select_dtypes(include=[np.number]).columns
        precision = self.config['numeric_precision']
        
        # Round numeric columns to specified precision
        df_norm[numeric_cols] = df_norm[numeric_cols].round(precision)
        
        # TODO: Implement scaling/normalization techniques as needed
        return df_norm
    
    def standardize_dates(self, df: pd.DataFrame, date_columns: List[str] = None) -> pd.DataFrame:
        """
        Standardize date columns to consistent format.
        
        Args:
            df: DataFrame containing date columns
            date_columns: List of column names containing dates
            
        Returns:
            DataFrame with standardized date columns
        """
        logger.info("Standardizing date columns")
        
        df_dates = df.copy()
        
        # Auto-detect date columns if not specified
        if date_columns is None:
            date_columns = []
            for col in df_dates.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    date_columns.append(col)
        
        date_format = self.config['date_format']
        
        for col in date_columns:
            if col in df_dates.columns:
                try:
                    df_dates[col] = pd.to_datetime(df_dates[col]).dt.strftime(date_format)
                    logger.info(f"Standardized date column: {col}")
                except Exception as e:
                    logger.warning(f"Could not standardize date column {col}: {e}")
        
        return df_dates
    
    def remove_duplicates(self, df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.
        
        Args:
            df: DataFrame to deduplicate
            subset: Columns to consider for identifying duplicates
            
        Returns:
            DataFrame with duplicates removed
        """
        logger.info("Removing duplicate rows")
        
        initial_count = len(df)
        df_dedup = df.drop_duplicates(subset=subset)
        final_count = len(df_dedup)
        
        logger.info(f"Removed {initial_count - final_count} duplicate rows")
        return df_dedup
    
    def validate_normalized_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the normalized dataset.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if validation passes, False otherwise
        """
        logger.info("Validating normalized data")
        
        # Check if dataframe is empty
        if df.empty:
            logger.error("Normalized dataset is empty")
            return False
        
        # Check for excessive missing values
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_pct > 0.1:  # 10% threshold
            logger.warning(f"High percentage of missing values: {missing_pct:.2%}")
        
        # TODO: Add more validation checks
        logger.info("Data validation completed successfully")
        return True
    
    def save_normalized_data(self, df: pd.DataFrame, output_path: Union[str, Path]) -> None:
        """
        Save normalized data to file.
        
        Args:
            df: Normalized DataFrame to save
            output_path: Path where normalized data should be saved
        """
        logger.info(f"Saving normalized data to {output_path}")
        
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if str(output_path).endswith('.csv'):
                df.to_csv(output_path, index=False)
            elif str(output_path).endswith('.json'):
                df.to_json(output_path, orient='records', indent=2)
            else:
                # Default to CSV
                df.to_csv(f"{output_path}.csv", index=False)
                
            logger.info(f"Successfully saved {len(df)} records")
            
        except Exception as e:
            logger.error(f"Error saving normalized data: {e}")
    
    def normalize_dataset(self, input_path: Union[str, Path], output_path: Union[str, Path]) -> bool:
        """
        Complete normalization pipeline for a dataset.
        
        Args:
            input_path: Path to raw data file
            output_path: Path to save normalized data
            
        Returns:
            True if normalization succeeded, False otherwise
        """
        logger.info(f"Starting data normalization pipeline")
        
        try:
            # Load raw data
            df = self.load_raw_data(input_path)
            if df.empty:
                logger.error("No data to normalize")
                return False
            
            # Apply normalization steps
            df = self.standardize_columns(df)
            df = self.handle_missing_values(df)
            df = self.normalize_numeric_columns(df)
            df = self.standardize_dates(df)
            df = self.remove_duplicates(df)
            
            # Validate normalized data
            if not self.validate_normalized_data(df):
                logger.error("Data validation failed")
                return False
            
            # Save normalized data
            self.save_normalized_data(df, output_path)
            
            logger.info("Data normalization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in normalization pipeline: {e}")
            return False


def main():
    """
    Main entry point for the data normalization script.
    """
    normalizer = DataNormalizer()
    
    # TODO: Configure input/output paths from command line or config file
    input_file = "raw_data.csv"
    output_file = "normalized_data.csv"
    
    success = normalizer.normalize_dataset(input_file, output_file)
    if success:
        logger.info("Normalization completed successfully")
    else:
        logger.error("Normalization failed")


if __name__ == "__main__":
    main()
