import logging
import os
import io
import mimetypes
from typing import List
import boto3
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse

from app.config import settings
from app.routers.auth import User, current_active_user

logger = logging.getLogger(__name__)

s3 = boto3.client(
    "s3",
    endpoint_url=settings.minio_endpoint,
    aws_access_key_id=settings.minio_access_key,
    aws_secret_access_key=settings.minio_secret_key,
    config=boto3.session.Config(s3={'addressing_style': 'path'})
)

router = APIRouter(
    prefix="/api/v1",
    tags=["files"],
    dependencies=[],
    responses={404: {"description": "Not found"}},
)

@router.get(
    "/files",
    summary="List MinIO Bucket Contents",
    description="""
    Retrieves a detailed list of objects stored within a specific MinIO bucket. 
    
    - **bucket_name**: Uses the `path` parameter to filter results by folder or directory structure.
    - **path**: Uses the `path` parameter to filter results by folder or directory structure.
    - **Authentication**: Requires a valid `current_active_user` session.
    - **S3 Compatibility**: Operates using standard S3-compatible prefix logic.
    """,
    response_description="A list of objects containing file names, sizes, and last modified timestamps."    
)
async def list_files(
    bucket_name: str = Query(..., description="Target bucket name"),
    path: str = Query(..., description="Folder path inside bucket"),
    tags: str = Query(None, description="Comma-separated tags to filter by (e.g. 'invoice,finance')"),
    current_active_user: User = Depends(current_active_user)
):
    try:
        files_metadata = []

        paginator = s3.get_paginator('list_objects_v2')

        if path and not path.endswith('/'):
            path += '/'
        path = path.replace(" ", "_")

        # Parse filter tags once, outside the loop
        filter_tags = set(t.strip() for t in tags.split(",")) if tags else None

        for page in paginator.paginate(Bucket=bucket_name, Prefix=path):
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                key = obj['Key']

                if key == path:
                    continue

                object_tags = []

                # Only fetch tags if a filter is requested or we always want to return them
                # Fetch tags for every object (for filtering and/or response)
                tag_response = s3.get_object_tagging(Bucket=bucket_name, Key=key)
                object_tags = [t["Key"] for t in tag_response.get("TagSet", [])]

                # Skip object if it doesn't match ALL requested filter tags
                if filter_tags and not filter_tags.issubset(set(object_tags)):
                    continue

                files_metadata.append({
                    "name": key.split('/')[-1],
                    "full_path": key,
                    "type": os.path.splitext(key)[1].replace('.', '') or 'file',
                    "last_modified": obj['LastModified'].isoformat(),
                    "size_bytes": obj['Size'],
                    "tags": object_tags
                })

        return {
            "status": "success",
            "count": len(files_metadata),
            "files": files_metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/files/upload",
    summary="Upload Files to MinIO",
    description="""
    Uploads a binary file to a specified destination within a MinIO bucket.
    
    - **Multipart Encoding**: The file must be sent as `multipart/form-data`.
    - **Path Resolution**: The `path` represents the object key prefix (folder structure).
    - **Overwrite Behavior**: If a file with the same name exists at the path, it will be overwritten.
    """,
    response_description="A JSON object containing the upload status and the final file URI."    
) 
async def upload_files(
    bucket_name: str = Query(..., description="Target bucket name"),
    path: str = Query(..., description="Folder path inside bucket"),
    files: List[UploadFile] = File(...), # Changed to List
    tags: str = Query(None, description="Comma-separated tags (e.g. 'invoice,finance,2024')"),
    current_active_user: User = Depends(current_active_user)
):
    uploaded_results = []
    errors = []

    # Parse tags once, shared across all files
    parsed_tags = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    tagging = {
        "TagSet": [{"Key": tag, "Value": "1"} for tag in parsed_tags]
    } if parsed_tags else None

    for file in files:
        try:                
            # Read file content
            file_content = await file.read()

            # Clean the path and combine with filename
            clean_path = path.strip("/").replace(" ", "_")
            object_key = f"{clean_path}/{file.filename}" if clean_path else file.filename

            # Upload to MinIO
            s3.put_object(
                Bucket=bucket_name,
                Key=object_key,
                Body=io.BytesIO(file_content),
                ContentType=file.content_type
            )

            try:
                if tagging:
                    s3.put_object_tagging(
                        Bucket=bucket_name,
                        Key=object_key,
                        Tagging=tagging
                    )
            except e:
                print(e)
                
            uploaded_results.append({
                "filename": file.filename,
                "status": "uploaded"
            })
            
        except Exception as e:
            errors.append({"filename": file.filename, "error": str(e)})

    if errors and not uploaded_results:
        raise HTTPException(status_code=500, detail={"message": "All uploads failed", "errors": errors})

    return {
        "uploaded": uploaded_results,
        "bucket": bucket_name,
        "tags": parsed_tags,        
        "failed": errors
    }
    
@router.delete(
    "/files",
    summary="Remove File from MinIO",
    description="""
    Permanently deletes a specific object from a MinIO bucket.
    
    - **Destructive Action**: This operation is irreversible. Once deleted, the file cannot be recovered unless versioning is enabled on the bucket.
    - **Object Key**: You must provide the full path including the filename (e.g., `uploads/2026/report.pdf`).
    - **Permissions**: Only the bucket owner or an authorized admin user can perform this action.
    """,
    response_description="Returns an empty body with HTTP 204 status on successful deletion."
)
def remove_file(
    bucket_name: str = Query(..., description="Target bucket name"),
    object_key: str = Query(..., description="The full path to the file (e.g., folder/document.pdf)"),
    current_active_user: User = Depends(current_active_user)
):
    try:
        # 1. Check if the object exists first (Optional but recommended for better error messages)
        s3.head_object(Bucket=bucket_name, Key=object_key)

        # 2. Perform the deletion
        s3.delete_object(Bucket=bucket_name, Key=object_key)
        
        return {
            "status": "success",
            "message": f"Object '{object_key}' deleted from '{bucket_name}'."
        }
    except s3.exceptions.ClientError as e:
        # If head_object fails with 404
        if e.response['Error']['Code'] == "404":
            raise HTTPException(status_code=404, detail=f"File '{object_key}' not found in bucket '{bucket_name}'.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
@router.get(
    "/files/download",
    summary="Download File from MinIO",
    description="""
    Fetches a file from the storage backend and streams it directly to the client.
    
    - **Streaming**: Large files are streamed to prevent high memory consumption on the server.
    - **Object Key**: Requires the full path to the file (e.g., `projects/dataset.zip`).
    - **Content-Type**: The API will attempt to serve the file with its original media type.
    """,
    response_description="A binary stream of the requested file."              
)
async def download_file(
    bucket_name: str,
    object_key: str,
    current_active_user: User = Depends(current_active_user)
):
    # 1. Definir un tipo por defecto al principio para evitar el "not defined"
    content_type = "application/octet-stream"
    
    try:
        # Intentar adivinar el tipo de contenido basándose en la extensión
        guessed_type, _ = mimetypes.guess_type(object_key)
        if guessed_type:
            content_type = guessed_type

        s3_response = s3.get_object(Bucket=bucket_name, Key=object_key)
        body = s3_response['Body']

        def generate_chunks():
            # Usamos un bloque 'with' para asegurar el cierre del stream
            with body as stream:
                while True:
                    # Leemos un pedazo del archivo
                    chunk = stream.read(1024 * 512) # 512KB
                    if not chunk:
                        break
                    yield chunk

        filename = object_key.split('/')[-1]

        return StreamingResponse(
            generate_chunks(),
            media_type=content_type,
            headers={
                # Las comillas dobles en filename ayudan con nombres con espacios
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(s3_response['ContentLength']),
                "Access-Control-Expose-Headers": "Content-Disposition, Content-Length"
            }
        )

    except s3.exceptions.ClientError as e:
        # If head_object fails with 404
        if e.response['Error']['Code'] == "404":
            raise HTTPException(status_code=404, detail=f"File '{object_key}' not found in bucket '{bucket_name}'.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))  