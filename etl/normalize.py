#!/usr/bin/env python3
"""
ETL Normalize Module - SQLite Schema and Data Loader
This module provides SQLite database schema definition and data loading functionality.
"""

import sqlite3
import pandas as pd
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SQLiteNormalizer:
    """SQLite schema and data loader for MoneyTaur pipeline."""
    
    def __init__(self, db_path: str = "moneytaur.db"):
        self.db_path = db_path
        logger.info(f"SQLiteNormalizer initialized with database: {db_path}")
    
    def create_schema(self) -> bool:
        """Create the database schema for financial data storage.
        
        Returns:
            True if schema creation successful, False otherwise
        """
        logger.info("Creating database schema")
        
        schema_sql = """
        CREATE TABLE IF NOT EXISTS financial_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            open_price DECIMAL(10, 4),
            high_price DECIMAL(10, 4),
            low_price DECIMAL(10, 4),
            close_price DECIMAL(10, 4),
            volume BIGINT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, date)
        );
        
        CREATE TABLE IF NOT EXISTS weekly_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol VARCHAR(20) NOT NULL,
            week_start_date DATE NOT NULL,
            week_end_date DATE NOT NULL,
            weekly_open DECIMAL(10, 4),
            weekly_close DECIMAL(10, 4),
            weekly_volume BIGINT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_financial_symbol_date ON financial_data(symbol, date);
        CREATE INDEX IF NOT EXISTS idx_weekly_symbol_date ON weekly_data(symbol, week_start_date);
        """
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript(schema_sql)
                logger.info("Database schema created successfully")
                return True
        except sqlite3.Error as e:
            logger.error(f"Error creating schema: {e}")
            return False
    
    def load_financial_data(self, data: pd.DataFrame) -> bool:
        """Load financial data into the database."""
        logger.info(f"Loading {len(data)} financial data records")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                data.to_sql('financial_data', conn, if_exists='append', index=False)
                logger.info("Financial data loaded successfully")
                return True
        except Exception as e:
            logger.error(f"Error loading financial data: {e}")
            return False
    
    def normalize_and_load(self, file_path: str) -> bool:
        """Complete ETL process: extract, normalize, and load data."""
        logger.info(f"Starting ETL process for {file_path}")
        
        try:
            if not Path(file_path).exists():
                logger.error(f"Source file not found: {file_path}")
                return False
                
            data = pd.read_csv(file_path)
            if data.empty:
                logger.warning("No data to process")
                return True
            
            # Basic normalization
            data.columns = data.columns.str.lower().str.replace(' ', '_')
            data = data.drop_duplicates()
            
            # Load data
            return self.load_financial_data(data)
            
        except Exception as e:
            logger.error(f"Error in ETL process: {e}")
            return False


def main():
    """Main entry point for the normalize module."""
    normalizer = SQLiteNormalizer("moneytaur_pipeline.db")
    
    if normalizer.create_schema():
        logger.info("Database schema ready")
    else:
        logger.error("Failed to create database schema")


if __name__ == "__main__":
    main()
