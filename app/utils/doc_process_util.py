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

def _download_from_s3(bucket_name: str, object_key: str) -> tuple[str, list[str]]:
    """
    Downloads a file from MinIO/S3 and saves it to a safe local temporary path.
    Returns a tuple of (absolute local path, list of tags).
    """
    s3_client = boto3.client(
        's3',
        endpoint_url=settings.minio_endpoint,
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
        config=Config(signature_version='s3v4')
    )

    _, ext = os.path.splitext(object_key.lower())

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
        local_path = temp_file.name

    try:
        s3_client.download_file(bucket_name, object_key, local_path)

        # Fetch object tags
        tag_response = s3_client.get_object_tagging(Bucket=bucket_name, Key=object_key)
        tags = [t["Key"] for t in tag_response.get("TagSet", [])]

        return local_path, tags
    except Exception as e:
        if os.path.exists(local_path):
            os.remove(local_path)
        raise e
    
def _ocr_image(file_path):
    """
    Extract text from image using Tesseract (tesseract) using spanish language 
    This language package must be install in host (tesseract-lang)
    """
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image, lang='spa')

    return text

def get_text_splitter():
    return text_splitter

def split_doc_by_chunks(bucket_name: str, object_key: str):
    sanitized_key = urllib.parse.unquote_plus(object_key).lstrip("/")
    _, ext = os.path.splitext(object_key.lower())

    docs = []
    local_file = None

    try:
        # Unpack path and tags together
        local_file, tags = _download_from_s3(bucket_name, sanitized_key)

        if ext in ["png", "jpg", "jpeg"]:
            text = _ocr_image(local_file)
            docs = [Document(page_content=text, metadata={"source": sanitized_key})]

        elif ext == "pdf":
            loader = PyPDFLoader(local_file)
            docs = loader.load()

        elif ext == "doc":
            doc = aw.Document(local_file)
            text_content = doc.get_text()
            docs = [Document(page_content=text_content, metadata={"source": sanitized_key})]

        else:
            loader = UnstructuredLoader(local_file)
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = sanitized_key

    except Exception as e:
        logger.error(f"Failed processing file '{sanitized_key}' from bucket '{bucket_name}': {e}")
    finally:
        if local_file and os.path.exists(local_file):
            try:
                os.remove(local_file)
            except Exception as cleanup_error:
                logger.warning(f"Failed to delete temp file {local_file}: {cleanup_error}")

    if not docs:
        logger.warning(f"No document content recovered for {sanitized_key}. Returning empty splits.")
        return []

    doc_splits = text_splitter.split_documents(docs)

    # Inject tags into every chunk's metadata
    for chunk in doc_splits:
        chunk.metadata["tags"] = tags

    return doc_splits