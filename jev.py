#!/usr/bin/env python3
"""
jev.py -- standard TypeSafe Jev quickstart: triage one customer support ticket.

Jev is a System One decision model: you send a state (the ticket) plus typed
questions and get probabilities back instead of prose.

Call it from your own main:

    from jev import triage, show
    response = triage("Our API integration is returning 500 errors.")
    print(response.choices["department"].choice)
    show(response)                      # human-readable summary

Setup:
    uv pip install typesafe-sdk python-dotenv
    echo 'jev_key=apikey_...' > .env    # TypeSafe key from console.typesafe.ai

Run:
    .venv/bin/python jev.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

MODEL = "jev-latest"

TICKET = (
    "Our API integration started returning 500 errors on every request about "
    "20 minutes ago, and we can't process any customer orders until this is fixed."
)

QUESTIONS = {
    "department": Choice(
        instructions="Which team should handle this",
        criteria={
            "billing": "Payment or subscription issues",
            "technical": "Bugs or integration problems",
            "sales": "Pricing or account questions",
        },
    ),
    "frustration": Score(
        instructions="How frustrated the customer appears",
        criteria=[
            "Calm, just stating facts",
            "Frustrated but civil",
            "Very angry, strong language",
        ],
    ),
    "is_urgent": Noul(
        instructions="The message conveys urgency or time-sensitivity",
    ),
}


def _api_key() -> str:
    """Read jev_key from .env, falling back to TYPESAFE_API_KEY."""
    load_dotenv(Path(__file__).with_name(".env"))
    key = os.getenv("jev_key") or os.getenv("TYPESAFE_API_KEY")
    if not key:
        raise SystemExit("No TypeSafe key found. Put jev_key=apikey_... in .env")
    return key


def ask(state: str, questions: dict, *, api_key: str | None = None, model: str = MODEL):
    """Send any state plus typed questions to Jev and return the SDK response."""
    with TypeSafeClient(model=model, api_key=api_key or _api_key()) as client:
        return client.system_one(state=state, questions=questions)


def triage(ticket: str = TICKET, *, api_key: str | None = None, model: str = MODEL):
    """Send one ticket to Jev and return the SDK response.

    The response exposes .model, .choices[name].choice/.probabilities,
    .scores[name].score and .nouls[name].noul.
    """
    return ask(ticket, QUESTIONS, api_key=api_key, model=model)


def show(response) -> None:
    """Print the typed answers in a readable form."""
    department = response.choices["department"]
    print("model:", response.model)
    print("department:", department.choice, department.probabilities)
    print("frustration:", response.scores["frustration"].score)
    print("is_urgent:", response.nouls["is_urgent"].noul)


def main() -> None:
    print(f"Ticket: {TICKET}\n")
    show(triage())


if __name__ == "__main__":
    main()
