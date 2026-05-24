import logging
import base64
import hashlib

import boto3

from fastapi import APIRouter, Depends, HTTPException, Query, status

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
    tags=["collections"],
    dependencies=[],
    responses={404: {"description": "Not found"}},
)

def add_content_md5(request, **kwargs):
    """Event hook that computes and injects Content-MD5 for DeleteObjects calls."""
    body = request.body
    if isinstance(body, str):
        body = body.encode('utf-8')
    md5 = base64.b64encode(hashlib.md5(body).digest()).decode('utf-8')
    request.headers['Content-MD5'] = md5

@router.get("/collections")
def get_collection_files(
    bucket_name: str = Query(..., description="Target bucket name"),
    current_active_user: User = Depends(current_active_user)
):
    # Storage for our stats
    path_stats = {}

    try:
        paginator = s3.get_paginator('list_objects_v2')
        # NO DELIMITER here: we want to see EVERYTHING inside the bucket
        page_iterator = paginator.paginate(Bucket=bucket_name)

        for page in page_iterator:
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                key = obj['Key']
                
                # 1. Identify the Top-Level Path
                if '/' in key:
                    path_prefix = key.split('/')[0]
                else:
                    path_prefix = "root"

                # 2. Initialize the dictionary for this path
                if path_prefix not in path_stats:
                    path_stats[path_prefix] = {"total_files": 0, "last_mod": None}

                # 3. Skip the "Folder Placeholder" objects (0-byte files ending in /)
                if key.endswith('/'):
                    continue

                # 4. It's a real file! Update the stats
                path_stats[path_prefix]["total_files"] += 1
                
                # Update date if this file is newer than what we have
                obj_date = obj['LastModified']
                if path_stats[path_prefix]["last_mod"] is None or obj_date > path_stats[path_prefix]["last_mod"]:
                    path_stats[path_prefix]["last_mod"] = obj_date

        # 5. Format results
        results = []
        for path, data in path_stats.items():
            results.append({
                "path": path.replace("_", " "),
                "total_files": data["total_files"],
                "last_file_creation": (
                    data["last_mod"].strftime("%Y-%m-%d %H:%M:%S") 
                    if data["last_mod"] is not None 
                    else None
                )
            })
            
        return results     
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
@router.post("/collections")
def create_collection(
    bucket_name: str = Query(..., description="Target bucket name"),
    path: str  = Query(..., description="Folder path inside bucket"),
    current_active_user: User = Depends(current_active_user)
):
    try:        
        # Ensure the path ends with a slash to represent a directory and don't have spaces between
        if not path.endswith('/'):
            path += '/'
    
        path = path.replace(" ", "_")

        s3.put_object(Bucket=bucket_name, Key=path)

        print(f"Key '{path}' created successfully in '{bucket_name}'.")

        return {
            "key": path,
            "bucket": bucket_name,
            "status": "created",
            "executed_by": current_active_user            
        }

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
@router.delete("/collections")
async def remove_collection(
    bucket_name: str = Query(..., description="Target bucket name"),
    path: str = Query(..., description="Folder path inside bucket"),
    current_active_user: User = Depends(current_active_user)
):
    try:
        s3.meta.events.register('before-send.s3.DeleteObjects', add_content_md5)
        
        # Ensure the path ends with a slash to avoid accidental partial matches
        path = path.replace(" ", "_")
        path = path.strip("/") + "/"    

        # 1. List all objects with the prefix
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=path)

        delete_count = 0
        for page in pages:
            if 'Contents' in page:
                # Prepare a list of keys to delete
                delete_keys = [{'Key': obj['Key']} for obj in page['Contents']]
                
                # 2. Batch delete (max 1000 per call)
                if delete_keys:
                    s3.delete_objects(
                        Bucket=bucket_name,
                        Delete={'Objects': delete_keys}
                    )
                    delete_count += len(delete_keys)

        if delete_count == 0:
            raise HTTPException(status_code=404, detail=f"No objects found in path '{path}'")
        
        return {
            "status": "success",
            "message": f"Successfully deleted {delete_count} objects from {path}",
            "executed_by": current_active_user
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))    
