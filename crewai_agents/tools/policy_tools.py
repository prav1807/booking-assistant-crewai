"""Policy retrieval and visa eligibility tools for the PolicyCrew."""

import csv
import json
import os
from typing import Any, Dict, List, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# ---------------------------------------------------------------------------
# Passport-index visa eligibility (structured lookup)
# ---------------------------------------------------------------------------

_visa_data: Optional[Dict[str, Dict[str, str]]] = None


def _load_visa_data() -> Dict[str, Dict[str, str]]:
    """Load the passport-index CSV into a nested dict: {passport: {destination: requirement}}."""
    global _visa_data
    if _visa_data is not None:
        return _visa_data

    path = os.path.join(DATA_DIR, "passport-index-tidy.csv")
    _visa_data = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            passport = row["Passport"].strip().lower()
            dest = row["Destination"].strip().lower()
            req = row["Requirement"].strip()
            _visa_data.setdefault(passport, {})[dest] = req

    return _visa_data


class VisaEligibilityInput(BaseModel):
    passport_country: str = Field(description="The traveller's passport country (e.g. 'Mauritius', 'United Kingdom')")
    destination_country: str = Field(description="The destination country (e.g. 'Japan', 'United States')")


class VisaEligibilityTool(BaseTool):
    name: str = "check_visa_eligibility"
    description: str = (
        "Check visa and entry requirements for a passport-destination pair. "
        "Returns whether the traveller needs a visa, can enter visa-free, "
        "or requires an eVisa/ETA. Uses the passport-index dataset."
    )
    args_schema: Type[BaseModel] = VisaEligibilityInput

    def _run(self, passport_country: str, destination_country: str) -> str:
        if not passport_country or not destination_country:
            return "ERROR: Both passport country and destination country are required."

        data = _load_visa_data()
        passport = passport_country.strip().lower()
        dest = destination_country.strip().lower()

        if passport not in data:
            return f"NO_DATA: No visa data found for passport country '{passport_country}'."

        if dest not in data[passport]:
            return f"NO_DATA: No visa data for '{passport_country}' passport travelling to '{destination_country}'."

        requirement = data[passport][dest]

        # Interpret common values from the dataset
        req_lower = requirement.lower()
        if req_lower in ("-1", "visa required"):
            status = "VISA_REQUIRED"
            explanation = f"Travellers holding a {passport_country} passport require a visa to enter {destination_country}."
        elif "visa free" in req_lower or req_lower.isdigit():
            days = requirement if requirement.isdigit() else "unknown"
            status = "VISA_FREE"
            explanation = f"Travellers holding a {passport_country} passport can enter {destination_country} visa-free for up to {days} days."
        elif "eta" in req_lower or "evisa" in req_lower or "e-visa" in req_lower:
            status = "EVISA_REQUIRED"
            explanation = f"Travellers holding a {passport_country} passport need an eVisa/ETA for {destination_country}. Apply online before travel."
        elif req_lower in ("-1", "-"):
            status = "VISA_REQUIRED"
            explanation = f"Travellers holding a {passport_country} passport require a visa to enter {destination_country}."
        else:
            status = requirement.upper()
            explanation = f"Entry requirement for {passport_country} passport to {destination_country}: {requirement}."

        return f"ELIGIBILITY: {status} | {explanation} | Source: passport-index dataset (Ilyankou, 2025)"


# ---------------------------------------------------------------------------
# Policy retrieval (Qdrant vector search with sentence-transformers)
# ---------------------------------------------------------------------------

_qdrant_collection: Optional[Any] = None
_embedding_model: Optional[Any] = None

COLLECTION_NAME = "policy_corpus"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _ensure_qdrant_collection():
    """Initialize in-memory Qdrant collection with the policy corpus."""
    global _qdrant_collection
    if _qdrant_collection is not None:
        return _qdrant_collection

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = QdrantClient(":memory:")
    model = _get_embedding_model()

    corpus_path = os.path.join(DATA_DIR, "policy_corpus.json")
    with open(corpus_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()
    dim = len(embeddings[0])

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i],
            payload={
                "doc_id": doc["id"],
                "source": doc["source"],
                "topic": doc["topic"],
                "text": doc["text"],
            },
        )
        for i, doc in enumerate(documents)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)

    _qdrant_collection = client
    return client


class PolicySearchInput(BaseModel):
    query: str = Field(description="The policy question to search for (e.g. 'What is the carry-on baggage limit?')")
    top_k: int = Field(default=3, description="Number of top passages to retrieve")


class PolicyRetrievalTool(BaseTool):
    name: str = "search_policy"
    description: str = (
        "Search the airline policy knowledge base for answers to questions about "
        "baggage rules, dangerous goods, disability assistance, fare rules, "
        "refunds, and ticket changes. Returns relevant policy passages with sources."
    )
    args_schema: Type[BaseModel] = PolicySearchInput

    def _run(self, query: str, top_k: int = 3) -> str:
        if not query or not query.strip():
            return "ERROR: No query provided."

        client = _ensure_qdrant_collection()
        model = _get_embedding_model()

        query_vector = model.encode(query, normalize_embeddings=True).tolist()

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
        ).points

        if not results:
            return "NO_RESULTS: No relevant policy passages found for this question."

        passages = []
        for i, hit in enumerate(results, 1):
            payload = hit.payload
            score = hit.score
            passages.append(
                f"[{i}] (relevance: {score:.3f}) [{payload['source']}]\n"
                f"Topic: {payload['topic']}\n"
                f"{payload['text']}"
            )

        return "POLICY_RESULTS:\n\n" + "\n\n".join(passages)
