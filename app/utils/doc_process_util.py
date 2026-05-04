from PIL import Image
import pytesseract
import tempfile
import boto3
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import S3FileLoader
from langchain_core.documents import Document

from app.config import settings
from app.utils.const import *

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,
                                               chunk_overlap=64,
                                               length_function=len)

def _download_from_s3(bucket_name, object_key):
    """
    Download file from MinIO/S3 into a temporary file
    """
    s3 = boto3.client(
        "s3",
        endpoint_url=settings.minio_endpoint,
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
    )

    tmp = tempfile.NamedTemporaryFile(delete=False)
    s3.download_fileobj(bucket_name, object_key, tmp)
    tmp.close()

    return tmp.name

def _ocr_image(file_path):
    """
    Extract text from image using Tesseract (tesseract) using spanish language 
    This language packahe must be install in host (tesseract-lang)
    """
    image = Image.open(file_path)
    #text = pytesseract.image_to_string(image)
    text = pytesseract.image_to_string(image, lang='spa')

    return text

def get_text_splitter():
    return text_splitter

def split_doc_by_chunks(bucket_name, object_key):
    # determine file type
    ext = object_key.lower().split(".")[-1]

    if ext in ["png", "jpg", "jpeg"]:
        # download image
        local_file = _download_from_s3(bucket_name, object_key)

        # OCR extraction
        text = _ocr_image(local_file)

        docs = [Document(page_content=text, metadata={"source": object_key})]
    else:
        # default loader for PDFs, docs, txt.
        loader = S3FileLoader(
            bucket_name,
            object_key,
            endpoint_url=settings.minio_endpoint,
            aws_access_key_id=settings.minio_access_key,
            aws_secret_access_key=settings.minio_secret_key
        )

        docs = loader.load()

    # split documents
    doc_splits = text_splitter.split_documents(docs)

    return doc_splits
