# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from service.assistant import AssistantService
from config import settings
from utils.logger import get_logger

from utils.performance_calc import Metrics
import time

from test.concurrency_test import run_test

metrics = Metrics()

logger = get_logger("fastapi_main")

assistant = AssistantService(
    data_path=settings.DEFAULT_DOCS_PATH,
    chroma_path=settings.DEFAULT_CHROMA_PATH
)

# Initialize the RAG components
assistant.initialize()

# Pydantic Models for API Contract
class QueryInput(BaseModel):
    """Schema for the incoming user query."""
    question: str

class SourceDocument(BaseModel):
    """Schema for a retrieved source document."""
    source: str
    page: str

class QueryResponse(BaseModel):
    """Schema for the final API response."""
    answer: str
    sources: List[SourceDocument]
    total_tokens: int

# FastAPI Application Instance
app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version
)

# 3. API Endpoints
@app.post("/query", response_model=QueryResponse)
async def handle_query(input: QueryInput):
    """
    Accepts a user question, performs RAG using the internal knowledge base, 
    and returns a grounded answer.
    """
    try:
        t0 = time.time()
        # Call the core RAG service function
        result = await assistant.query(input.question)
        ms = (time.time() - t0) * 1000
        metrics.total.record(ms)
        # FastAPI handles serializing the dict to the Pydantic response model
        return result
    except Exception as err:
        # Handle cases where the RAG chain failed to initialize or execute
        raise HTTPException(status_code=500, detail=str(err))

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "assistant_ready": assistant.chain is not None
    }

@app.post("/reindex")
async def reindex():
    try:
        result = await assistant.reindex()
        assistant.initialize()
        return result
    except Exception as err:
        # Failed to update DB
        logger.error(f"Error @ reindex | Error ==> {err}")
        raise HTTPException(status_code=500, detail=str(err))

@app.get("/metrics")
def get_metrics():
    return metrics.stats()

@app.post("/reset_metrics")
def reset_metrics():
    metrics.reset_all()
    return {"message": "Metrics reset"}

@app.post("/concurrency_test")
async def concurrency_test():
    return await run_test()

# Run the Application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)
