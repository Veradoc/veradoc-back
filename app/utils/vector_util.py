import threading
import logging
from datetime import timedelta
import numpy as np
import pyarrow as pa
import requests
import torch

import lancedb
from lancedb.pydantic import LanceModel, Vector

from sentence_transformers import CrossEncoder

from app.utils.const import *
from app.config import settings

logger = logging.getLogger(__name__)

db = None
table = None

class DocsModel(LanceModel):
    parent_source: str
    source: str
    text: str    
    tags: list[str]
    owner_id: str
    vector: Vector(EMBEDDINGS_DIM, pa.float16()) # type: ignore

_reranker_model: CrossEncoder | None = None
_reranker_lock = threading.Lock()

def _get_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"

def get_reranker() -> CrossEncoder:
    global _reranker_model

    if _reranker_model is None:
        with _reranker_lock:
            if _reranker_model is None:
                device = _get_best_device()
                
                print(f"[reranker] Loading {RERANKER_MODEL} on {device}...")

                _reranker_model = CrossEncoder(
                    RERANKER_MODEL,
                    device=device,
                    trust_remote_code=True)
                
                print(f"[reranker] Model ready on {device}")                
    return _reranker_model

def get_db():
    global db

    if db is None:
        db = lancedb.connect(
            "s3://warehouse/v-db/",
            read_consistency_interval=timedelta(seconds=5)
        )

    #db.drop_table("docs")  # replace with your DOCS_TABLE value
    #print("Table dropped.")

def get_or_create_table():
    global table

    # Lazy db client creation for LanceDB initialization
    get_db()

    if table is None and DOCS_TABLE not in list(db.table_names()):
        return db.create_table(DOCS_TABLE, schema=DocsModel)
    if table is None:
        table = db.open_table(DOCS_TABLE)

    return table

def get_embedding(text):
    resp = requests.post(
        settings.ollama_host + "/api/embeddings",
        json={
            "model": EMBEDDING_MODEL,
            "prompt": text
        }
    )

    return np.array(resp.json()["embedding"][:EMBEDDINGS_DIM], dtype=np.float16)

def search(query, top_vectors, tags: list[str] = None):
    query_embedding = get_embedding(f"{EMBEDDING_QUERY_PREFIX}: {query}")
    search_query = get_or_create_table().search(query_embedding).metric("cosine")

    if tags:
        # LanceDB SQL filter: check each tag is present in the array column
        tag_conditions = " AND ".join(f"array_has(tags, '{tag}')" for tag in tags)
        search_query = search_query.where(tag_conditions)

    return search_query.limit(top_vectors)

def search_reranker(query, top_vectors, top_reranker_vectors, tags: list[str] = None):
    TASK_DESCRIPTION = "Given a question, retrieve relevant passages that answer the question"
    query_embedding = get_embedding(f"Instruct: {TASK_DESCRIPTION}\nQuery: {query}")
    #query_embedding = get_embedding(f"{EMBEDDING_QUERY_PREFIX}: {query}")
    search_query = get_or_create_table().search(query_embedding).metric("cosine")

    if tags:
        # LanceDB SQL filter: check each tag is present in the array column
        tag_conditions = " AND ".join(f"array_has(tags, '{tag}')" for tag in tags)
        search_query = search_query.where(tag_conditions)

    candidates = search_query.limit(top_reranker_vectors).to_list()

    if not candidates:
        return []

    texts = [doc["text"] for doc in candidates]
    pairs = [(query, text) for text in texts]

    try:
        scores = get_reranker().predict(pairs)  # ndarray of float32

        for doc, score in zip(candidates, scores):
            doc["rerank_score"] = float(score)

        return sorted(candidates, key=lambda d: d["rerank_score"], reverse=True)[:top_vectors]

    except Exception as e:
        print(f"[reranker] CrossEncoder failed, falling back to LanceDB order: {e}")

        return candidates[:top_vectors]