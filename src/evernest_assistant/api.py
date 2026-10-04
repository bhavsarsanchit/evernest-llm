"""HTTP API for the listing agent. The contract is openapi.yaml."""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from evernest_assistant.agent import ask
from evernest_assistant.index import index_documents
from evernest_assistant.spec import load_spec

app = FastAPI(title="Evernest listing agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    agent_id: str = Field(examples=["A-01"])
    listing_id: str = Field(examples=["L-1001"])
    question: str
    model: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    document_id: str
    page: int
    listing_id: str
    sub_category: str


class AccessInfo(BaseModel):
    can_read_listing: bool
    can_read_documents: bool
    reason: str
    role: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    access: AccessInfo
    model: str
    temperature: float
    max_tokens: int
    top_k: int
    refused: bool
    tokens: dict


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/spec")
def spec() -> dict:
    current = load_spec()
    return {
        "model": current.model,
        "base_url": current.base_url,
        "temperature": current.temperature,
        "max_tokens": current.max_tokens,
        "top_k": current.top_k,
        "system": current.system,
    }


@app.post("/index")
def rebuild_index() -> dict:
    return index_documents()


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(body: AskRequest) -> dict:
    return ask(
        agent_id=body.agent_id,
        listing_id=body.listing_id,
        question=body.question,
        model=body.model,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        top_k=body.top_k,
    )
