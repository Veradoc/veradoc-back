import os
import logging
import tempfile
import urllib.parse

import boto3
import pytesseract
from PIL import Image
from botocore.config import Config
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.auto import partition

from app.config import settings
from app.utils.const import *

try:
    import aspose.words as aw
    ASPOSE_AVAILABLE = True
except ImportError:
    ASPOSE_AVAILABLE = False

logger = logging.getLogger(__name__)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500,
    chunk_overlap=400,
    length_function=len
)


def get_text_splitter():
    return text_splitter


def _download_from_s3(bucket_name: str, object_key: str) -> tuple[str, list[str]]:
    """
    Downloads a file from MinIO/S3 into a safe local temporary path.
    Returns a tuple of (absolute local path, list of tags).
    """
    s3 = boto3.client(
        "s3",
        endpoint_url=settings.minio_endpoint,
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
        config=Config(signature_version='s3v4')
    )

    _, ext = os.path.splitext(object_key.lower())

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        local_path = tmp.name

    try:
        s3.download_file(bucket_name, object_key, local_path)

        tag_response = s3.get_object_tagging(Bucket=bucket_name, Key=object_key)

        tag_set = tag_response.get("TagSet", [])
        tags = [t["Value"] for t in tag_set if t["Key"] != "owner_id"]
        owner_id = next(
            (t["Value"] for t in tag_set if t["Key"] == "owner_id"),
        None
)

        return local_path, tags, owner_id
    except Exception as e:
        if os.path.exists(local_path):
            os.remove(local_path)
        raise e


def _ocr_image(file_path: str) -> str:
    """
    Extract text from image using Tesseract with Spanish language.
    Requires tesseract-lang package installed on host.
    """
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image, lang='spa')
    return text


def split_doc_by_chunks(bucket_name: str, object_key: str) -> list:
    # Decode special characters and drop leading slashes
    sanitized_key = urllib.parse.unquote_plus(object_key).lstrip("/")
    _, ext = os.path.splitext(sanitized_key.lower())

    docs = []
    local_file = None

    try:
        local_file, tags, owner_id = _download_from_s3(bucket_name, sanitized_key)

        if ext in [".png", ".jpg", ".jpeg"]:
            text = _ocr_image(local_file)
            docs = [Document(page_content=text, metadata={"source": sanitized_key})]

        elif ext == ".pdf":
            loader = PyPDFLoader(local_file)
            docs = loader.load()

        elif ext == ".doc":
            if not ASPOSE_AVAILABLE:
                raise ImportError("aspose.words is required to process .doc files")
            doc = aw.Document(local_file)
            text_content = doc.get_text()
            docs = [Document(page_content=text_content, metadata={"source": sanitized_key})]

        else:
            # Fallback for .docx, .txt, etc.
            loader = UnstructuredFileLoader(local_file)
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
        logger.warning(f"No content recovered for '{sanitized_key}'. Returning empty splits.")
        return []

    # Chunking the documents
    doc_splits = text_splitter.split_documents(docs)

    # Inject tags and owner_id into every chunk metadata
    for chunk in doc_splits:
        chunk.metadata["tags"] = list(tags)
        chunk.metadata["owner_id"] = owner_id

    return doc_splits