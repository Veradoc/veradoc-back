import logging
import json
import multiprocessing
import urllib.parse
import pandas as pd
import boto3
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import APIRouter, FastAPI, Request, BackgroundTasks
from pydantic import BaseModel
from contextlib import asynccontextmanager

from app.config import settings
from app.utils.const import *
from app.utils.doc_process_util import split_doc_by_chunks
from app.utils.vector_util import get_embedding, get_or_create_table
from app.utils.websocket_manager import ws_manager
from app.routers.auth import engine, Base

logger = logging.getLogger(__name__)

s3 = boto3.client(
    "s3",
    endpoint_url=settings.minio_endpoint,
    aws_access_key_id=settings.minio_access_key,
    aws_secret_access_key=settings.minio_secret_key,
    config=boto3.session.Config(s3={'addressing_style': 'path'})
)

context_df = []

# multiprocessing queues to handle data
add_data_queue = multiprocessing.Queue()
delete_data_queue = multiprocessing.Queue()

def add_vector_job():
    data = []
    table = get_or_create_table()

    while not add_data_queue.empty():
        item = add_data_queue.get()
        data.append(item)

    if len(data) > 0:
        df = pd.DataFrame(data)
        table.add(df)
        table.compact_files()
        print(f"Total Rows Added: {len(table.to_pandas())}")

def delete_vector_job():
    table = get_or_create_table()
    source_data = []
    while not delete_data_queue.empty():
        item = delete_data_queue.get()
        source_data.append(item)
    if len(source_data) > 0:
        filter_data = ", ".join([f'"{d}"' for d in source_data])
        table.delete(f'source IN ({filter_data})')
        table.compact_files()
        table.cleanup_old_versions()
        print(f"Total Rows Deleted: {len(table.to_pandas())}")

scheduler = BackgroundScheduler()

scheduler.add_job(add_vector_job, 'interval', seconds=10)
scheduler.add_job(delete_vector_job, 'interval', seconds=10)

@asynccontextmanager
async def lifespan_llm(app: FastAPI):
    # This creates the users table in your SQLite/Postgres DB if it doesn't exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # --- APP STARTUP ---
    get_or_create_table()
    if not scheduler.running:
        scheduler.start()
    yield

    # --- SHUTDOWN ---
    scheduler.shutdown()

router = APIRouter(
    prefix="/api/v1",
    tags=["minio"],
    dependencies=[],
    responses={404: {"description": "Not found"}},
)

class WebhookData(BaseModel):
    data: dict

@router.post("/document/notification")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    json_data = await request.json()
    print(json.dumps(json_data, indent=2))

    if json_data["EventName"] == "s3:ObjectCreated:Put":
        print("New object created!")
        background_tasks.add_task(create_object_task, json_data)
    if json_data["EventName"] == "s3:ObjectRemoved:Delete":
        print("Object deleted!")
        background_tasks.add_task(delete_object_task, json_data)

    return {"status": "success"}

@router.post("/metadata/notification")
async def receive_metadata_webhook(request: Request, background_tasks: BackgroundTasks):
    json_data = await request.json()
    print(json.dumps(json_data, indent=2))
    if json_data["EventName"] == "s3:ObjectCreated:Put":
        print("New Metadata created!")
        background_tasks.add_task(create_metadata_task, json_data)
    if json_data["EventName"] == "s3:ObjectRemoved:Delete":
        print("Metadata deleted!")
        background_tasks.add_task(delete_metadata_task, json_data)

    return {"status": "success"}

def create_object_task(json_data):
    for record in json_data["Records"]:
        bucket_name = record["s3"]["bucket"]["name"]
        object_key = urllib.parse.unquote(record["s3"]["object"]["key"])

        # 1. Detect if it's a directory
        # In S3, directories are objects ending in '/'
        if object_key.endswith('/'):
            print(f"Skipping bucket key creation: {object_key}")
            continue

        # 2. Safety check: Ensure it's not a zero-byte placeholder
        size = record["s3"]["object"].get("size", 0)
        if size == 0:
            print(f"Skipping empty object or placeholder: {object_key}")
            continue

        try:
            chunks = split_doc_by_chunks(bucket_name, object_key)

            for i, chunk in enumerate(chunks):
                # Prepare the JSON data
                chunk_data = chunk.json()

                print(chunk_data)

                # Define the destination Key (Note: We remove the 'warehouse/' prefix from the key
                # IF 'warehouse' is your bucket name. If 'warehouse' is a folder inside a bucket, keep it.)
                # For this example, I'll assume 'warehouse' is the BUCKET_NAME.
                target_bucket = "warehouse"
                target_key = f"{METADATA_PREFIX}/{bucket_name}/{object_key}/chunk_{i:05d}.json"

                try:
                    s3.put_object(
                        Bucket=target_bucket,
                        Key=target_key,
                        Body=chunk_data,
                        ContentType="application/json"
                    )
                    print(f"Successfully saved chunk {i} to {target_key}")

                except Exception as e:
                    print(f"Failed to save chunk {i}: {e}")
                    # Depending on your needs, you might want to raise the error to stop processing
                    # raise e

        except Exception as e:
            print(f"Error processing {object_key}: {e}")

    return "Task completed!"

def delete_object_task(json_data) -> str:
    TARGET_BUCKET = "warehouse"
    errors_found = []

    for record in json_data["Records"]:
        bucket_name = record["s3"]["bucket"]["name"]
        object_key = urllib.parse.unquote(record["s3"]["object"]["key"])

        # Skip directory markers (S3 "folders" ending with /)
        if object_key.endswith('/'):
            print(f"Skipping directory marker: {object_key}")
            continue

        # Build the prefix pointing to the chunked metadata folder
        # e.g. "metadata/custom-corpus/path/to/file.pdf/"
        prefix = f"{METADATA_PREFIX}/{bucket_name}/{object_key}/"

        print(f"Deleting chunks under prefix: s3://{TARGET_BUCKET}/{prefix}")

        try:
            # Paginate through all objects under this prefix
            paginator = s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=TARGET_BUCKET, Prefix=prefix)

            delete_batch = [
                {'Key': obj['Key']}
                for page in pages
                for obj in page.get('Contents', [])  # .get avoids KeyError on empty pages
            ]

            if not delete_batch:
                print(f"No chunks found to delete under: {prefix}")
                continue

            # delete_objects supports up to 1000 keys per call
            BATCH_LIMIT = 1000
            for i in range(0, len(delete_batch), BATCH_LIMIT):
                chunk = delete_batch[i:i + BATCH_LIMIT]

                response = s3.delete_objects(
                    Bucket=TARGET_BUCKET,       # ✓ delete from the same bucket you listed
                    Delete={
                        'Objects': chunk,
                        'Quiet': True           # only returns errors, not every deleted key
                    }
                    # No ChecksumAlgorithm — boto3 sends Content-MD5 automatically
                )

                # delete_objects returns HTTP 200 even on partial failures — always check
                failed = response.get('Errors', [])
                if failed:
                    for err in failed:
                        msg = f"Failed to delete {err['Key']}: [{err['Code']}] {err['Message']}"
                        print(msg)
                        errors_found.append(msg)
                else:
                    print(f"Deleted {len(chunk)} chunks from batch starting at index {i}")

        except Exception as e:
            print(f"Error during recursive delete of {object_key}: {e}")

    if errors_found:
        return f"Task completed with {len(errors_found)} error(s):\n" + "\n".join(errors_found)

    return "Task completed!"

def create_metadata_task(json_data):
    for record in json_data["Records"]:
        bucket_name = record["s3"]["bucket"]["name"]
        object_key = urllib.parse.unquote(record["s3"]["object"]["key"])

        print(bucket_name, object_key)

        try:
            # 1. Get the object from S3
            response = s3.get_object(Bucket=bucket_name, Key=object_key)

            # 2. Read the body stream and decode it
            # 'Body' is a StreamingBody object, .read() gets the bytes
            data = response['Body'].read().decode('utf-8')

            # 3. Parse JSON
            chunk_json = json.loads(data)

            print(chunk_json)

            # 4. Process Embeddings
            text_to_embed = f"{EMBEDDING_DOCUMENT_PREFIX}: {chunk_json['page_content']}"
            embeddings = get_embedding(text_to_embed)

            # 5. Add to Queue
            add_data_queue.put({
                "text": chunk_json["page_content"],
                "parent_source": chunk_json.get("metadata", {}).get("source", ""),
                "source": f"{bucket_name}/{object_key}",
                "vector": embeddings,
                "tags": list(chunk_json.get("metadata", {}).get("tags", []))          
            })

        except s3.exceptions.NoSuchKey:
            print(f"Error: The object {object_key} does not exist.")
        except Exception as e:
            print(f"Error processing {object_key}: {e}")

    # 6. Get tags for this object
    tagging_response = s3.get_object_tagging(Bucket=bucket_name, Key=object_key)

    # 7. Extract owner_id from TagSet
    tag_set = tagging_response.get('TagSet', [])
    owner_id = next(
        (tag['Value'] for tag in tag_set if tag['Key'] == 'owner_id'),
        None  # default if tag not found
    )

    # 8. send websocket event when save embbedings
    ws_manager.send_to_user(
        owner_id,
        {
            "event": "doc.ingested",
            "doc_id": object_key,
            "filename": f"{bucket_name}/{object_key}",
            #"chunks": len(chunks),
            "status": "ready"
        }
    )
        
    return "Task Completed!"

def delete_metadata_task(json_data):
    for record in json_data["Records"]:
        bucket_name = record["s3"]["bucket"]["name"]
        object_key = urllib.parse.unquote(record["s3"]["object"]["key"])
        delete_data_queue.put(f"{bucket_name}/{object_key}")

    return "Task completed!"

def clear_events():
    global context_df

    context_df = []

    return context_df
