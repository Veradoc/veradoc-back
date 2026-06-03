import logging
from datetime import timedelta
import numpy as np
import pyarrow as pa
import requests

import ollama
import lancedb
from lancedb.pydantic import LanceModel, Vector

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
    vector: Vector(EMBEDDINGS_DIM, pa.float16()) # type: ignore

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
    resp = requests.post(settings.ollama_host + "/api/embeddings",
                         json={"model": EMBEDDING_MODEL, "prompt": text})

    # Log the real dimension once to make sure your config is correct
    #print(f"Model {EMBEDDING_MODEL} returned {len(resp.json()["embedding"])} dims")

    return np.array(resp.json()["embedding"][:EMBEDDINGS_DIM], dtype=np.float16)

def search(query, limit=5, tags: list[str] = None):
    query_embedding = get_embedding(f"{EMBEDDING_QUERY_PREFIX}: {query}")
    
    search_query = get_or_create_table().search(query_embedding).metric("cosine")

    if tags:
        # LanceDB SQL filter: check each tag is present in the array column
        tag_conditions = " AND ".join(f"array_has(tags, '{tag}')" for tag in tags)
        
        search_query = search_query.where(tag_conditions)

    return search_query.limit(limit)

def search_reranker(query, limit=5, tags: list[str] = None):
    query_embedding = get_embedding(f"{EMBEDDING_QUERY_PREFIX}: {query}")
    
    search_query = get_or_create_table().search(query_embedding).metric("cosine")

    if tags:
        # LanceDB SQL filter: check each tag is present in the array column
        tag_conditions = " AND ".join(f"array_has(tags, '{tag}')" for tag in tags)
        
        search_query = search_query.where(tag_conditions)

    canditatos_db = search_query.limit(TOP_RERANKER_VECTORS)

    if not canditatos_db:
        return []
    
    # 3. Extraemos los textos para el Reranker de Ollama
    documentos_texto = [doc["text"] for doc in canditatos_db]

    try:
        # 4. Llamada al Reranker (A Ollama no le importan los vectores aquí, solo el texto plano)
        rerank_response = ollama.post(
            model=RERANKER_MODEL,
            json={
                "query": query,
                "documents": documentos_texto
            }
        )
        
        resultados_ordenados = rerank_response.json().get("results", [])
        
        # 5. Reordenamos los objetos devueltos por LanceDB
        documentos_rerankeados = []
        for res in resultados_ordenados:
            idx = res["index"]
            doc_original = canditatos_db[idx]
            doc_original["rerank_score"] = res["relevance_score"]
            documentos_rerankeados.append(doc_original)
            
        return documentos_rerankeados[:limit]

    except Exception as e:
        print(f"Error en el Reranker, devolviendo fallback de LanceDB: {e}")
        return canditatos_db[:limit]    