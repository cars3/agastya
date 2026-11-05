#!/usr/bin/env python3
"""One decision endpoint, two backends: TypeSafe Jev (cloud) or the local
unsloth FastDecisionModel (a fine-tuned Qwen3.5-2B).

Both are System One decision models: you send a state plus typed questions and
get probabilities back instead of prose. This file wraps them behind a single
schema so callers cannot tell which backend answered.

    POST /decide
    {
      "state": "I was double charged on invoice #4092. Please refund my card.",
      "backend": "cloud",                  # or "local"
      "questions": {
        "intent":  {"type": "choice", "instructions": "Classify the request.",
                    "criteria": {"billing": "...", "technical": "...", "general": "..."}},
        "urgency": {"type": "score",  "instructions": "How urgent is this?",
                    "criteria": ["Can wait", "Soon", "Immediately"]},
        "urgent":  {"type": "noul", "instructions": "Does this need escalation?"}
      }
    }

Always answers with the same shape, whichever backend ran:

    {
      "backend": "local",
      "model": "unsloth_Qwen3.5-2B__project-agastya-rag_1793806165",
      "latency_ms": 412.7,
      "answers": {
        "intent": {"type": "choice", "answer": "billing", "confidence": 0.5266,
                   "probabilities": {"billing": 0.5266, "technical": 0.0257, "general": 0.4477}},
        "urgency": {"type": "score", "answer": 1, "confidence": 0.61,
                    "probabilities": {"0": 0.21, "1": 0.61, "2": 0.18},
                    "legend": {"0": "Can wait", "1": "Soon", "2": "Immediately"}},
        "urgent": {"type": "noul", "answer": false, "confidence": 0.5446,
                   "probabilities": {"true": 0.4554, "false": 0.5446}}
      }
    }

Run:
    /usr/bin/python3.14 decision_model.py --serve --port 8000
    /usr/bin/python3.14 decision_model.py --backend local --state "..."

Setup:
    cloud  needs jev_key=apikey_... in .env (or TYPESAFE_API_KEY)
    local  needs the checkpoint beside this file (see LOCAL_MODEL_DIR below)
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Literal

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typesafe_sdk import TypeSafeClient

BASE = Path(__file__).resolve().parent
CLOUD_MODEL = "jev-latest"
LOCAL_MODEL_DIR = Path(
    os.environ.get("DECISION_MODEL_DIR")
    or BASE / "unsloth_Qwen3.5-2B__project-agastya-rag_1793806165"
)

Backend = Literal["cloud", "local"]


# --------------------------------------------------------------------------
# Request schema. These dicts are passed through to either backend almost
# verbatim: the TypeSafe SDK's question TypedDicts and FastDecisionModel.predict
# both use {type, instructions, criteria}, so only validation happens here.
# --------------------------------------------------------------------------
class Question(BaseModel):
    type: Literal["choice", "score", "noul"]
    instructions: str = ""
    criteria: dict[str, str] | list[str] | None = None

    @field_validator("criteria")
    @classmethod
    def _criteria_match_type(cls, value, info):
        qtype = info.data.get("type")
        if qtype == "choice" and not isinstance(value, dict):
            raise ValueError("choice questions need criteria as {option: description}")
        if qtype == "score" and not isinstance(value, list):
            raise ValueError("score questions need criteria as an ordered list of levels")
        if qtype == "noul" and value is not None:
            raise ValueError("noul questions take no criteria")
        return value


class DecideRequest(BaseModel):
    state: str = Field(min_length=1)
    questions: dict[str, Question] = Field(min_length=1)
    backend: Backend = "local"
    model: str | None = None  # cloud only; ignored by the local backend


# --------------------------------------------------------------------------
# One canonical answer shape for both backends.
# --------------------------------------------------------------------------
def _round(probabilities: dict[Any, Any]) -> dict[str, float]:
    """JSON-safe probabilities: string keys, rounded, sum preserved."""
    return {str(key): round(float(value), 4) for key, value in probabilities.items()}


def _normalize(kind: str, raw: dict[str, Any], criteria: Any) -> dict[str, Any]:
    """Turn one backend's answer into the shared shape.

    Cloud answers arrive as pydantic models dumped to dicts, local ones as plain
    dicts. Both expose probabilities over every option; the field names differ
    only in that a choice carries `choice` and a score carries `score` + legend.
    """
    probabilities = _round(raw.get("probabilities") or {})

    if kind == "choice":
        answer = raw.get("choice", raw.get("answer"))
        confidence = raw.get("confidence")
        if confidence is None:
            confidence = probabilities.get(str(answer), 0.0)
        return {
            "type": "choice",
            "answer": answer,
            "confidence": round(float(confidence), 4),
            "probabilities": probabilities,
        }

    if kind == "noul":
        answer = raw.get("answer")
        if answer is None:
            answer = float(raw.get("noul", 0.0)) >= 0.5
        confidence = raw.get("confidence")
        if confidence is None:
            confidence = max(
                probabilities.get("true", 0.0), probabilities.get("false", 0.0)
            )
        return {
            "type": "noul",
            "answer": bool(answer),
            "confidence": round(float(confidence), 4),
            "probabilities": probabilities,
        }

    # score
    answer = raw.get("answer", raw.get("score"))
    legend = raw.get("legend")
    if legend is None and isinstance(criteria, list):
        legend = {str(i): level for i, level in enumerate(criteria)}
    elif legend is not None and not isinstance(legend, dict):
        legend = {str(i): level for i, level in enumerate(legend)}
    confidence = raw.get("confidence")
    if confidence is None:
        confidence = probabilities.get(str(answer), 0.0)
    result = {
        "type": "score",
        "answer": int(answer) if answer is not None else None,
        "confidence": round(float(confidence), 4),
        "probabilities": probabilities,
    }
    if legend:
        result["legend"] = {str(key): value for key, value in legend.items()}
    return result


# --------------------------------------------------------------------------
# Cloud backend: TypeSafe Jev.
# --------------------------------------------------------------------------
def _api_key() -> str:
    load_dotenv(BASE / ".env")
    key = os.getenv("jev_key") or os.getenv("TYPESAFE_API_KEY")
    if not key:
        raise HTTPException(
            status_code=503,
            detail="Cloud backend unavailable: put jev_key=apikey_... in .env "
            "(or export TYPESAFE_API_KEY).",
        )
    return key


def predict_cloud(req: DecideRequest) -> dict[str, Any]:
    questions = {
        name: question.model_dump(exclude_none=True)
        for name, question in req.questions.items()
    }
    with TypeSafeClient(model=req.model or CLOUD_MODEL, api_key=_api_key()) as client:
        response = client.system_one(state=req.state, questions=questions)

    return {
        "backend": "cloud",
        "model": response.model,
        "answers": {
            name: _normalize(
                answer.type,
                answer.model_dump(),
                req.questions[name].criteria,
            )
            for name, answer in response.answers.items()
        },
    }


# --------------------------------------------------------------------------
# Local backend: unsloth FastDecisionModel over the fine-tuned checkpoint.
# Loaded on first local request so the API starts (and serves cloud traffic)
# on a box where the GPU is busy or absent.
# --------------------------------------------------------------------------
_local_model = None
_local_tokenizer = None
_local_model_name = None
_local_lock = threading.Lock()


def _load_local():
    global _local_model, _local_tokenizer, _local_model_name
    with _local_lock:
        if _local_model is not None:
            return _local_model, _local_tokenizer
        if not LOCAL_MODEL_DIR.is_dir():
            raise HTTPException(
                status_code=503,
                detail=f"Local backend unavailable: checkpoint not found at {LOCAL_MODEL_DIR}",
            )
        try:
            from unsloth import FastDecisionModel
        except Exception as error:  # noqa: BLE001 - surfaced verbatim to the caller
            raise HTTPException(
                status_code=503,
                detail=f"Local backend unavailable: cannot import unsloth ({error})",
            ) from error

        try:
            model, tokenizer = FastDecisionModel.from_pretrained(
                model_name=str(LOCAL_MODEL_DIR),
                max_seq_length=2048,
                # The checkpoint was trained 4-bit; fp32 is ~8.5 GB and will OOM
                # on an 8 GB card before inference starts.
                load_in_4bit=True,
            )
            FastDecisionModel.for_inference(model)
        except Exception as error:  # noqa: BLE001
            raise HTTPException(
                status_code=503,
                detail=f"Local backend unavailable: model load failed ({error})",
            ) from error

        _local_model, _local_tokenizer = model, tokenizer
        _local_model_name = LOCAL_MODEL_DIR.name
        return _local_model, _local_tokenizer


def predict_local(req: DecideRequest) -> dict[str, Any]:
    from unsloth import FastDecisionModel

    model, tokenizer = _load_local()
    questions = {
        name: question.model_dump(exclude_none=True)
        for name, question in req.questions.items()
    }
    with _local_lock:  # one forward pass at a time; the model is shared
        raw = FastDecisionModel.predict(
            model=model, tokenizer=tokenizer, state=req.state, questions=questions
        )

    return {
        "backend": "local",
        "model": _local_model_name,
        "answers": {
            name: _normalize(
                req.questions[name].type,
                answer,
                req.questions[name].criteria,
            )
            for name, answer in raw.items()
        },
    }


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
app = FastAPI(title="Agastya decision model", version="1.0.0")

PREDICTORS = {"cloud": predict_cloud, "local": predict_local}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "backends": {
            "cloud": {
                "ready": bool(os.getenv("jev_key") or os.getenv("TYPESAFE_API_KEY")),
                "model": CLOUD_MODEL,
            },
            "local": {
                "loaded": _local_model is not None,
                "model": LOCAL_MODEL_DIR.name,
                "checkpoint_present": LOCAL_MODEL_DIR.is_dir(),
            },
        },
    }


@app.post("/decide")
def decide(req: DecideRequest) -> dict[str, Any]:
    started = time.perf_counter()
    result = PREDICTORS[req.backend](req)
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
DEMO_STATE = (
    "I was double charged on invoice #4092. Can you please issue a refund to my credit card?"
)

DEMO_QUESTIONS = {
    "intent": Question(
        type="choice",
        instructions="Classify the primary request category.",
        criteria={
            "billing": "Inquiries regarding charges, payments, or refunds.",
            "technical": "Issues relating to product bugs or server outages.",
            "general": "General questions or feedback.",
        },
    ),
    "urgent": Question(
        type="noul",
        instructions="Does this request need immediate escalation?",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend", choices=["cloud", "local"], default="local")
    parser.add_argument("--state", default=DEMO_STATE)
    parser.add_argument("--questions", type=Path, help="JSON file of questions")
    args = parser.parse_args()

    if args.serve:
        uvicorn.run(app, host=args.host, port=args.port)
        return

    questions = (
        json.loads(args.questions.read_text(encoding="utf-8"))
        if args.questions
        else {name: q.model_dump(exclude_none=True) for name, q in DEMO_QUESTIONS.items()}
    )
    request = DecideRequest(state=args.state, questions=questions, backend=args.backend)
    print(json.dumps(decide(request), indent=2))


if __name__ == "__main__":
    main()
