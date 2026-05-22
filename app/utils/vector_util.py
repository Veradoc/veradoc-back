from datetime import timedelta
import numpy as np
import pyarrow as pa
import requests
import lancedb
from lancedb.pydantic import LanceModel, Vector

from app.utils.const import *
from app.config import settings

db = None
table = None

class DocsModel(LanceModel):
    parent_source: str
    source: str
    text: str
    vector: Vector(EMBEDDINGS_DIM, pa.float16()) # type: ignore

def get_db():
    global db

    if db is None:
        db = lancedb.connect(
            "s3://warehouse/v-db/",
            read_consistency_interval=timedelta(seconds=5)
        )

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
    resp = requests.post(settings.ollama_host + "/api/embeddings",
                         json={"model": EMBEDDING_MODEL, "prompt": text})

    # Log the real dimension once to make sure your config is correct
    #print(f"Model {EMBEDDING_MODEL} returned {len(resp.json()["embedding"])} dims")

    return np.array(resp.json()["embedding"][:EMBEDDINGS_DIM], dtype=np.float16)

def search(query, limit=5):
    query_embedding = get_embedding(f"{EMBEDDING_QUERY_PREFIX}: {query}")
    res = get_or_create_table().search(query_embedding).metric("cosine").limit(limit)

    return res
