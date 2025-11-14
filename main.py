# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from service.assistant import AssistantService
from config import settings

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
        # Call the core RAG service function
        result = await assistant.query(input.question)
        
        # FastAPI handles serializing the dict to the Pydantic response model
        return result
    except Exception as e:
        # Handle cases where the RAG chain failed to initialize or execute
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "assistant_ready": assistant.chain is not None
    }

# Run the Application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)
