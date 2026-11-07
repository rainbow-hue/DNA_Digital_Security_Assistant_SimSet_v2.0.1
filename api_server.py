# -*- coding: utf-8 -*-
"""Mobile-app API for the DNA Digital Security Assistant (v3).

    pip install fastapi uvicorn numpy            (google-genai optional)
    python -m assistant.train --final            (once: builds assistant/artifacts/dsa_model.*)
    uvicorn api_server:app --host 0.0.0.0 --port 8000

POST /api/chat     one chat turn. Send back the `session_id` you receive.
POST /api/reset    forget a session.
GET  /api/health

Every reply has `reply_type`:
  "question"         show `display_text`; the next user message is the answer
  "consent_request"  show `display_text` with Allow / Don't allow buttons
                     (send `consent: {"granted": [...]}` or just "yes"/"no" as text)
  "final"            show `display_text`; `answer` holds the assignment's submission
                     object (risk_level, primary_action, steps, escalation ...)
After a final answer the user can keep typing ("I already paid, what now?"):
the assistant re-assesses with the new information.
"""
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from assistant.chat import ChatSessions
from assistant.engine import Assistant, DEFAULT_MODEL

GEMINI = None
if os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY_1"):
    from assistant.gemini_tier import GeminiTier
    GEMINI = GeminiTier.from_env()

ENGINE = Assistant(os.environ.get("DSA_MODEL", DEFAULT_MODEL), gemini=GEMINI)
SESSIONS = ChatSessions(ENGINE)

app = FastAPI(title="Digital Security Assistant API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])


class ConsentIn(BaseModel):
    granted: List[str] = []
    refused: List[str] = []


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    reset: bool = False
    language: Optional[str] = Field(None, description="'fi' | 'en' | 'sv'; detected when omitted")
    journey: Optional[str] = None
    audience: Optional[str] = Field(None, description="'consumer' | 'small_business'")
    signals: Optional[Dict[str, Any]] = Field(None, description="Operator-side signals, same shape as the case 'signals'")
    pre_granted: Optional[List[str]] = Field(None, description="Consents the user already gave in settings")
    consent: Optional[ConsentIn] = None


class ChatResponse(BaseModel):
    session_id: str
    reply_type: str
    display_text: str
    language: str
    journey: Optional[str] = None
    question: Optional[Dict[str, Any]] = None
    consent_request: Optional[Dict[str, Any]] = None
    answer: Optional[Dict[str, Any]] = None
    tier_used: Optional[str] = None
    safety_rules: Optional[List[str]] = None


class ResetRequest(BaseModel):
    session_id: str


@app.get("/api/health")
def health():
    return {"status": "ok", "gemini_tier": GEMINI is not None, "languages": ["fi", "en", "sv"]}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty.")
    return SESSIONS.handle(req.message, session_id=req.session_id, language=req.language, journey=req.journey,
                           audience=req.audience, signals=req.signals, pre_granted=req.pre_granted,
                           consent=req.consent.model_dump() if req.consent else None, reset=req.reset)


@app.post("/api/reset")
def reset(req: ResetRequest):
    SESSIONS.reset(req.session_id)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
