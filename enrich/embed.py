#!/usr/bin/env python3
"""
Data Embedding and Enrichment Module for MoneyTaur Pipeline

This module handles the creation of embeddings for financial data using OpenAI's
embedding models, enabling semantic search and similarity analysis.
"""

import logging
import os
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import json
import asyncio
from datetime import datetime

try:
    import openai
except ImportError:
    print("Warning: openai package not installed. Install with: pip install openai")
    openai = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataEmbedder:
    """
    Handles data embedding and enrichment using OpenAI embeddings.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the DataEmbedder.
        
        Args:
            config: Configuration dictionary for embedding parameters
        """
        self.config = config or {
            'embedding_model': 'text-embedding-3-small',
            'max_tokens': 8192,
            'batch_size': 100,
            'embedding_dimension': 1536
        }
        
        # Initialize OpenAI client
        self.client = None
        self._init_openai_client()
        
        logger.info("DataEmbedder initialized")
    
    def _init_openai_client(self) -> None:
        """
        Initialize OpenAI client with API key.
        """
        if openai is None:
            logger.error("OpenAI package not installed")
            return
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("OPENAI_API_KEY environment variable not set")
            return
        
        try:
            self.client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
    
    def load_normalized_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load normalized data from file.
        
        Args:
            file_path: Path to the normalized data file
            
        Returns:
            DataFrame containing normalized data
        """
        logger.info(f"Loading normalized data from {file_path}")
        try:
            if str(file_path).endswith('.csv'):
                return pd.read_csv(file_path)
            elif str(file_path).endswith('.json'):
                return pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def prepare_text_for_embedding(self, row: pd.Series, text_columns: List[str]) -> str:
        """
        Prepare text content from multiple columns for embedding.
        
        Args:
            row: DataFrame row containing data
            text_columns: List of column names to combine for embedding
            
        Returns:
            Combined text string ready for embedding
        """
        text_parts = []
        
        for col in text_columns:
            if col in row.index and pd.notna(row[col]):
                text_parts.append(f"{col}: {str(row[col])}")
        
        # Add numeric columns as context
        for col, value in row.items():
            if col not in text_columns and pd.notna(value):
                if isinstance(value, (int, float)):
                    text_parts.append(f"{col}: {value}")
        
        return " | ".join(text_parts)
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a batch of texts using OpenAI API.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not self.client:
            logger.error("OpenAI client not initialized")
            return []
        
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.config['embedding_model']
            )
            
            embeddings = [data.embedding for data in response.data]
            logger.info(f"Created embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return []
    
    def create_embeddings_for_dataframe(
        self, 
        df: pd.DataFrame, 
        text_columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Create embeddings for all rows in a DataFrame.
        
        Args:
            df: DataFrame containing data to embed
            text_columns: List of columns to use for embedding text
            
        Returns:
            DataFrame with added embedding column
        """
        logger.info(f"Creating embeddings for {len(df)} rows")
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return df
        
        # Auto-detect text columns if not specified
        if text_columns is None:
            text_columns = []
            for col in df.columns:
                if df[col].dtype == 'object':  # String columns
                    text_columns.append(col)
        
        if not text_columns:
            logger.warning("No text columns found for embedding")
            return df
        
        # Prepare texts for embedding
        texts = []
        for _, row in df.iterrows():
            text = self.prepare_text_for_embedding(row, text_columns)
            texts.append(text)
        
        # Create embeddings in batches
        all_embeddings = []
        batch_size = self.config['batch_size']
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.create_embeddings_batch(batch_texts)
            
            if not batch_embeddings:
                logger.error(f"Failed to create embeddings for batch {i//batch_size + 1}")
                # Add empty embeddings as fallback
                empty_embedding = [0.0] * self.config['embedding_dimension']
                batch_embeddings = [empty_embedding] * len(batch_texts)
            
            all_embeddings.extend(batch_embeddings)
        
        # Add embeddings to DataFrame
        df_enriched = df.copy()
        df_enriched['embedding'] = all_embeddings
        df_enriched['embedding_text'] = texts
        df_enriched['embedding_created_at'] = datetime.now().isoformat()
        
        logger.info(f"Successfully created embeddings for {len(all_embeddings)} rows")
        return df_enriched
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def find_similar_items(
        self, 
        query_embedding: List[float], 
        df: pd.DataFrame, 
        top_k: int = 5
    ) -> pd.DataFrame:
        """
        Find the most similar items to a query embedding.
        
        Args:
            query_embedding: Embedding vector to find similarities for
            df: DataFrame containing items with embeddings
            top_k: Number of most similar items to return
            
        Returns:
            DataFrame containing top-k most similar items
        """
        if 'embedding' not in df.columns:
            logger.error("No embedding column found in DataFrame")
            return pd.DataFrame()
        
        similarities = []
        for _, row in df.iterrows():
            if isinstance(row['embedding'], list):
                similarity = self.calculate_similarity(query_embedding, row['embedding'])
                similarities.append(similarity)
            else:
                similarities.append(0.0)
        
        df_with_similarity = df.copy()
        df_with_similarity['similarity_score'] = similarities
        
        # Sort by similarity and return top-k
        top_similar = df_with_similarity.nlargest(top_k, 'similarity_score')
        
        logger.info(f"Found top {len(top_similar)} similar items")
        return top_similar
    
    def save_enriched_data(
        self, 
        df: pd.DataFrame, 
        output_path: Union[str, Path], 
        include_embeddings: bool = True
    ) -> None:
        """
        Save enriched data with embeddings to file.
        
        Args:
            df: DataFrame with embeddings to save
            output_path: Path where enriched data should be saved
            include_embeddings: Whether to include embedding vectors in output
        """
        logger.info(f"Saving enriched data to {output_path}")
        
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            df_to_save = df.copy()
            
            # Optionally exclude embeddings to reduce file size
            if not include_embeddings and 'embedding' in df_to_save.columns:
                df_to_save = df_to_save.drop(columns=['embedding'])
            
            if str(output_path).endswith('.csv'):
                # Convert embedding lists to strings for CSV
                if 'embedding' in df_to_save.columns:
                    df_to_save['embedding'] = df_to_save['embedding'].apply(
                        lambda x: json.dumps(x) if isinstance(x, list) else x
                    )
                df_to_save.to_csv(output_path, index=False)
                
            elif str(output_path).endswith('.json'):
                df_to_save.to_json(output_path, orient='records', indent=2)
                
            else:
                # Default to JSON for complex data
                df_to_save.to_json(f"{output_path}.json", orient='records', indent=2)
            
            logger.info(f"Successfully saved {len(df_to_save)} enriched records")
            
        except Exception as e:
            logger.error(f"Error saving enriched data: {e}")
    
    def enrich_dataset(
        self, 
        input_path: Union[str, Path], 
        output_path: Union[str, Path], 
        text_columns: List[str] = None
    ) -> bool:
        """
        Complete enrichment pipeline for a dataset.
        
        Args:
            input_path: Path to normalized data file
            output_path: Path to save enriched data
            text_columns: Columns to use for creating embeddings
            
        Returns:
            True if enrichment succeeded, False otherwise
        """
        logger.info(f"Starting data enrichment pipeline")
        
        try:
            # Load normalized data
            df = self.load_normalized_data(input_path)
            if df.empty:
                logger.error("No data to enrich")
                return False
            
            # Create embeddings
            df_enriched = self.create_embeddings_for_dataframe(df, text_columns)
            
            if 'embedding' not in df_enriched.columns:
                logger.error("Failed to create embeddings")
                return False
            
            # Save enriched data
            self.save_enriched_data(df_enriched, output_path)
            
            logger.info("Data enrichment completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in enrichment pipeline: {e}")
            return False


def main():
    """
    Main entry point for the data enrichment script.
    """
    embedder = DataEmbedder()
    
    # TODO: Configure input/output paths from command line or config file
    input_file = "normalized_data.csv"
    output_file = "enriched_data.json"
    
    success = embedder.enrich_dataset(input_file, output_file)
    if success:
        logger.info("Enrichment completed successfully")
    else:
        logger.error("Enrichment failed")


if __name__ == "__main__":
    main()
