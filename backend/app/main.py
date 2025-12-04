from fastapi import FastAPI
from pydantic import BaseModel
import os

from ingest import ingest_url

app = FastAPI()

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class IngestRequest(BaseModel):
    url: str
    collection: str = "tutor_docs"


@app.post("/ingest")
def ingest(request: IngestRequest):
    result = ingest_url(
        url=request.url,
        qdrant_url=QDRANT_URL,
        api_key=OPENAI_API_KEY,
        collection_name=request.collection
    )
    return result


@app.get("/")
def root():
    return {"status": "backend online"}
