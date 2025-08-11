#!/usr/bin/env python3
"""
FastAPI Application for MoneyTaur Pipeline

This module provides a REST API interface for the MoneyTaur data pipeline,
including endpoints for data ingestion, processing, and semantic search using
OpenAI embeddings.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
except ImportError:
    print("Warning: fastapi not installed. Install with: pip install fastapi uvicorn")
    FastAPI = None

try:
    from pydantic import BaseModel, Field
except ImportError:
    print("Warning: pydantic not installed. Install with: pip install pydantic")
    BaseModel = None

try:
    import pandas as pd
except ImportError:
    print("Warning: pandas not installed. Install with: pip install pandas")
    pd = None

try:
    import openai
except ImportError:
    print("Warning: openai not installed. Install with: pip install openai")
    openai = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MoneyTaur Pipeline API",
    description="REST API for MoneyTaur financial data pipeline with OpenAI embeddings",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
) if FastAPI else None

if app:
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Pydantic models for request/response schemas
if BaseModel:
    class HealthResponse(BaseModel):
        status: str = "healthy"
        timestamp: datetime
        version: str = "1.0.0"
    
    class IngestionRequest(BaseModel):
        source: str = Field(..., description="Data source identifier")
        start_date: Optional[str] = Field(None, description="Start date for data ingestion (YYYY-MM-DD)")
        end_date: Optional[str] = Field(None, description="End date for data ingestion (YYYY-MM-DD)")
        lookback_weeks: int = Field(1, description="Number of weeks to look back")
    
    class ProcessingRequest(BaseModel):
        input_file: str = Field(..., description="Path to input data file")
        output_file: str = Field(..., description="Path to output data file")
        process_type: str = Field(..., description="Type of processing: 'normalize' or 'enrich'")
    
    class SearchRequest(BaseModel):
        query: str = Field(..., description="Search query text")
        top_k: int = Field(5, description="Number of results to return")
        dataset: Optional[str] = Field(None, description="Dataset to search in")
    
    class SearchResult(BaseModel):
        id: str
        score: float
        content: Dict[str, Any]
        metadata: Optional[Dict[str, Any]] = None
    
    class SearchResponse(BaseModel):
        query: str
        results: List[SearchResult]
        total_results: int
        processing_time_ms: float

# Global variables for pipeline components
embedding_service = None
data_store = {}


def init_openai_client():
    """
    Initialize OpenAI client for embeddings.
    """
    global embedding_service
    
    if not openai:
        logger.warning("OpenAI not available")
        return None
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.warning("OPENAI_API_KEY environment variable not set")
        return None
    
    try:
        embedding_service = openai.OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized successfully")
        return embedding_service
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return None


def create_embedding(text: str) -> List[float]:
    """
    Create embedding for given text using OpenAI API.
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector or empty list if failed
    """
    if not embedding_service:
        logger.error("Embedding service not initialized")
        return []
    
    try:
        response = embedding_service.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
        return []


if app:
    @app.on_event("startup")
    async def startup_event():
        """
        Initialize services on application startup.
        """
        logger.info("Starting MoneyTaur Pipeline API")
        init_openai_client()
        logger.info("API startup complete")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """
        Cleanup on application shutdown.
        """
        logger.info("Shutting down MoneyTaur Pipeline API")
    
    @app.get("/", response_model=Dict[str, str])
    async def root():
        """
        Root endpoint with basic API information.
        """
        return {
            "message": "MoneyTaur Pipeline API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health"
        }
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """
        Health check endpoint.
        """
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            version="1.0.0"
        )
    
    @app.post("/api/v1/ingest")
    async def trigger_ingestion(
        request: IngestionRequest,
        background_tasks: BackgroundTasks
    ):
        """
        Trigger data ingestion process.
        
        This endpoint starts a background task to ingest data from the specified source.
        """
        try:
            # TODO: Implement actual ingestion logic
            logger.info(f"Triggering ingestion for source: {request.source}")
            
            # Add background task for ingestion
            background_tasks.add_task(
                run_ingestion_task,
                request.source,
                request.lookback_weeks
            )
            
            return {
                "message": "Ingestion started",
                "source": request.source,
                "status": "processing",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error triggering ingestion: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v1/process")
    async def process_data(
        request: ProcessingRequest,
        background_tasks: BackgroundTasks
    ):
        """
        Process data (normalize or enrich).
        
        This endpoint processes data based on the specified type.
        """
        try:
            if request.process_type not in ["normalize", "enrich"]:
                raise HTTPException(
                    status_code=400, 
                    detail="process_type must be 'normalize' or 'enrich'"
                )
            
            logger.info(f"Processing data: {request.process_type}")
            
            # Add background task for processing
            background_tasks.add_task(
                run_processing_task,
                request.input_file,
                request.output_file,
                request.process_type
            )
            
            return {
                "message": f"Data processing ({request.process_type}) started",
                "input_file": request.input_file,
                "output_file": request.output_file,
                "status": "processing",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v1/search", response_model=SearchResponse)
    async def semantic_search(request: SearchRequest):
        """
        Perform semantic search using embeddings.
        
        This endpoint creates an embedding for the query and finds similar items.
        """
        start_time = datetime.now()
        
        try:
            # Create embedding for query
            query_embedding = create_embedding(request.query)
            if not query_embedding:
                raise HTTPException(
                    status_code=500, 
                    detail="Failed to create query embedding"
                )
            
            # TODO: Implement actual search logic with vector similarity
            # For now, return mock results
            mock_results = [
                SearchResult(
                    id=f"item_{i}",
                    score=0.9 - (i * 0.1),
                    content={"title": f"Sample result {i}", "description": f"Mock result {i} for query: {request.query}"},
                    metadata={"source": "mock_data"}
                ) for i in range(min(request.top_k, 3))
            ]
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return SearchResponse(
                query=request.query,
                results=mock_results,
                total_results=len(mock_results),
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/datasets")
    async def list_datasets():
        """
        List available datasets.
        """
        # TODO: Implement actual dataset listing
        return {
            "datasets": [
                {"name": "financial_data", "records": 1000, "last_updated": "2025-08-11T16:00:00Z"},
                {"name": "market_indices", "records": 500, "last_updated": "2025-08-11T15:30:00Z"}
            ],
            "total_datasets": 2
        }
    
    @app.get("/api/v1/status/{task_id}")
    async def get_task_status(task_id: str):
        """
        Get status of a background task.
        """
        # TODO: Implement actual task status tracking
        return {
            "task_id": task_id,
            "status": "completed",
            "progress": 100,
            "message": "Task completed successfully",
            "timestamp": datetime.now().isoformat()
        }


async def run_ingestion_task(source: str, lookback_weeks: int):
    """
    Background task for data ingestion.
    """
    logger.info(f"Running ingestion task for source: {source}")
    # TODO: Implement actual ingestion logic
    # This would import and run the weekly_index_ingest.py module
    await asyncio.sleep(2)  # Simulate processing time
    logger.info(f"Ingestion task completed for source: {source}")


async def run_processing_task(input_file: str, output_file: str, process_type: str):
    """
    Background task for data processing.
    """
    logger.info(f"Running {process_type} task: {input_file} -> {output_file}")
    # TODO: Implement actual processing logic
    # This would import and run normalize.py or embed.py modules
    await asyncio.sleep(3)  # Simulate processing time
    logger.info(f"Processing task ({process_type}) completed")


if __name__ == "__main__":
    import uvicorn
    
    if not app:
        print("FastAPI not available. Install dependencies: pip install fastapi uvicorn")
        exit(1)
    
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
