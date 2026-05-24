import logging
import tempfile
import urllib.parse

import boto3
from botocore.client import Config

from PIL import Image
import pytesseract
import aspose.words as aw

from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from app.config import settings
from app.utils.const import *

logger = logging.getLogger(__name__)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,
                                               chunk_overlap=64,
                                               length_function=len)

def _download_from_s3_v2(bucket_name: str, object_key: str) -> str:
    """
    Downloads a file from MinIO/S3 and saves it to a safe local temporary path.
    Returns the absolute path to the local file.
    """
    # 1. Initialize the S3 client for MinIO
    s3_client = boto3.client(
        's3',
        endpoint_url=settings.minio_endpoint,
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
        config=Config(signature_version='s3v4') # Ensures modern signature compatibility
    )
    
    # 2. Extract the file extension (e.g., '.pdf') to preserve format mapping
    _, ext = os.path.splitext(object_key.lower())
    
    # 3. Create a unique local file in the system's temp directory.
    # By using NamedTemporaryFile, Python generates a random name (no spaces!)
    # delete=False keeps the file on disk so LangChain loaders can open it next.
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
        local_path = temp_file.name
        
    try:
        # 4. Stream the file directly from MinIO into our safe local path
        s3_client.download_file(bucket_name, object_key, local_path)
        return local_path
    except Exception as e:
        # If the download fails, clean up the empty file we just created
        if os.path.exists(local_path):
            os.remove(local_path)
        raise e
    
def _ocr_image(file_path):
    """
    Extract text from image using Tesseract (tesseract) using spanish language 
    This language packahe must be install in host (tesseract-lang)
    """
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image, lang='spa')

    return text

def get_text_splitter():
    return text_splitter

def split_doc_by_chunks(bucket_name: str, object_key: str):
    # 1. Decode special characters/spaces and drop leading slashes
    sanitized_key = urllib.parse.unquote_plus(object_key).lstrip("/")
    
    # Get file type based on the cleaned key
    ext = sanitized_key.lower().split(".")[-1]

    docs = []
    local_file = None

    try:
        # 2. Download ALL files locally first using your working S3 helper.
        # This bypasses LangChain's internal temp_dir space-handling bugs.
        local_file = _download_from_s3_v2(bucket_name, sanitized_key)
        
        if ext in ["png", "jpg", "jpeg"]:
            # OCR extraction
            text = _ocr_image(local_file)
            docs = [Document(page_content=text, metadata={"source": sanitized_key})]
            
        elif ext == "pdf":
            # Direct PDF loader on our safely managed local file path
            loader = PyPDFLoader(local_file)
            docs = loader.load()

        elif ext == "doc":            
            # Convert the raw binary .doc directly into plain markdown text
            # 1. Load the binary file directly from disk
            doc = aw.Document(local_file)
                
            # 2. Extract the clean text directly into memory
            text_content = doc.get_text()

            docs = [Document(page_content=text_content, metadata={"source": sanitized_key})]
            
        else:
            # Fallback loader for docx, txt, etc.
            loader = UnstructuredLoader(local_file)
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = sanitized_key

    except Exception as e:
        logger.error(f"Failed processing file '{sanitized_key}' from bucket '{bucket_name}': {e}")
    finally:
        # 3. Clean up the local file after reading it to save disk space
        if local_file and os.path.exists(local_file):
            try:
                os.remove(local_file)
            except Exception as cleanup_error:
                logger.warning(f"Failed to delete temp file {local_file}: {cleanup_error}")

    # Guard rail: Only split if we successfully read content
    if not docs:
        logger.warning(f"No document content recovered for {sanitized_key}. Returning empty splits.")
        return []

    # split documents
    doc_splits = text_splitter.split_documents(docs)

    return doc_splits
