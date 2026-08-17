"""Centralised LLM configuration for all CrewAI crews.

Change the model here once — all crews pick it up automatically.
"""

import os

from crewai import LLM

# Read from env or default to llama3.1
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))


def build_llm() -> LLM:
    """Create the shared Ollama LLM instance."""
    return LLM(
        model=f"ollama/{OLLAMA_MODEL}",
        base_url=OLLAMA_BASE_URL,
        temperature=OLLAMA_TEMPERATURE,
        timeout=OLLAMA_TIMEOUT,
        max_retries=2,
    )
