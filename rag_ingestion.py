import boto3
import os
import io
import httpx
import hashlib
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Optional
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings

load_dotenv()
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')

def get_api_client():
    """Get HTTP client for API calls"""
    return httpx.Client(base_url=API_BASE_URL, timeout=30.0)

def get_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION', 'us-east-1')
    )

def list_s3_files(bucket_name: str = None, prefix: str = "") -> List[Dict[str, str]]:
    if bucket_name is None:
        bucket_name = os.getenv('AWS_S3_BUCKET')
        if not bucket_name:
            raise ValueError("Bucket name must be provided or set in AWS_S3_BUCKET environment variable")
    
    s3_client = get_s3_client()
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix
        )
        
        files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                files.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat(),
                    'etag': obj['ETag'].strip('"')
                })
        
        return files
    
    except Exception as e:
        print(f"Error listing files from S3: {str(e)}")
        raise

def download_file_from_s3(bucket_name: str, file_key: str) -> bytes:
    s3_client = get_s3_client()
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        return response['Body'].read()
    except Exception as e:
        print(f"Error downloading file {file_key}: {str(e)}")
        raise

def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

def get_pdf_text_from_s3(bucket_name: str, file_key: str) -> str:
    file_content = download_file_from_s3(bucket_name, file_key)
    return extract_text_from_pdf(file_content)

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_bedrock_embeddings():
    return BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        region_name=os.getenv('AWS_REGION', 'us-east-1')
    )

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    embeddings_model = get_bedrock_embeddings()
    embeddings = embeddings_model.embed_documents(chunks)
    return embeddings

def precheck_s3_file(bucket_name: str, file_info: Dict) -> Dict:
    """Check if S3 file should be downloaded based on fingerprint"""
    try:
        with get_api_client() as client:
            response = client.post("/precheck/s3", json={
                "bucket": bucket_name,
                "key": file_info["key"],
                "etag": file_info["etag"],
                "size": file_info["size"],
                "last_modified": file_info["last_modified"],
            })
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"Precheck failed for {file_info['key']}, falling back to download: {e}")
        return {"should_download": True}

def check_document_exists(content: str) -> Dict:
    """Check if document with given content checksum already exists"""
    norm = content.strip()
    sha = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    
    try:
        with get_api_client() as client:
            response = client.get("/documents/checksum-exists", params={"sha": sha})
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"Checksum check failed: {e}")
        return {"exists": False}

def get_or_create_embedding_model(name: str = "amazon.titan-embed-text-v1", 
                                 provider: str = "bedrock", embedding_dim: int = 1536,
                                 distance_metric: str = "cosine", version: str = "v1") -> str:
    """Get or create embedding model via API"""
    with get_api_client() as client:
        response = client.post("/embedding-models", json={
            "name": name,
            "provider": provider,
            "embedding_dim": embedding_dim,
            "distance_metric": distance_metric,
            "version": version
        })
        response.raise_for_status()
        return response.json()["id"]

def create_ingestion_run(source_label: str = None, notes: str = None) -> str:
    """Create a new ingestion run via API"""
    with get_api_client() as client:
        response = client.post("/ingestion-runs", json={
            "source_label": source_label,
            "notes": notes
        })
        response.raise_for_status()
        return response.json()["id"]

def update_ingestion_run_status(run_id: str, status: str, notes: str = None):
    """Update ingestion run status via API"""
    with get_api_client() as client:
        response = client.put(f"/ingestion-runs/{run_id}", json={
            "status": status,
            "notes": notes
        })
        response.raise_for_status()

def get_or_create_source(kind: str, uri: str, bucket: str = None, 
                        s3_key: str = None, s3_etag: str = None, 
                        content_hash: str = None, last_modified: str = None,
                        metadata: Dict = None) -> str:
    """Get or create source via API"""
    with get_api_client() as client:
        response = client.post("/sources", json={
            "kind": kind,
            "uri": uri,
            "bucket": bucket,
            "s3_key": s3_key,
            "s3_etag": s3_etag,
            "content_hash": content_hash,
            "last_modified": last_modified,
            "metadata": metadata or {}
        })
        response.raise_for_status()
        return response.json()["id"]

def create_document(source_id: str, ingestion_run_id: str, 
                   external_id: str = None, uri: str = None, mime_type: str = None,
                   title: str = None, author: str = None, language: str = None,
                   content: str = None, custom_tags: List[str] = None,
                   metadata: Dict = None) -> str:
    """Create document via API"""
    with get_api_client() as client:
        response = client.post("/documents", json={
            "source_id": source_id,
            "ingestion_run_id": ingestion_run_id,
            "external_id": external_id,
            "uri": uri,
            "mime_type": mime_type,
            "title": title,
            "author": author,
            "language": language,
            "content": content,
            "custom_tags": custom_tags,
            "metadata": metadata or {}
        })
        response.raise_for_status()
        return response.json()["id"]

def create_chunks(document_id: str, chunks: List[str], 
                 metadata_list: List[Dict] = None) -> List[str]:
    """Create chunks via API"""
    with get_api_client() as client:
        response = client.post("/chunks", json={
            "document_id": document_id,
            "chunks": chunks,
            "metadata_list": metadata_list
        })
        response.raise_for_status()
        return response.json()["chunk_ids"]

def create_chunk_embeddings(chunk_ids: List[str], embeddings: List[List[float]], 
                           model_id: str) -> List[str]:
    """Create chunk embeddings via API"""
    with get_api_client() as client:
        response = client.post("/embeddings", json={
            "chunk_ids": chunk_ids,
            "embeddings": embeddings,
            "model_id": model_id
        })
        response.raise_for_status()
        return response.json()["embedding_ids"]


def process_s3_file_to_api(bucket_name: str, file_info: Dict, 
                          ingestion_run_id: str, model_id: str) -> bool:
    """Process a single S3 file and store it via API with incremental checks"""
    file_key = file_info['key']
    
    try:
        print(f"Processing: {file_key}")
        

        text = get_pdf_text_from_s3(bucket_name, file_key)
        if not text or text.startswith("Error extracting text"):
            print(f"Failed to extract text from {file_key}")
            return False
        
        print(f"Extracted text length: {len(text)} characters")
        

        checksum_check = check_document_exists(text)
        if checksum_check.get("exists"):
            print(f"SKIP chunk+embed (same checksum): {file_key}")
            return True
        

        source_uri = f"s3://{bucket_name}/{file_key}"
        source_metadata = {
            "size": file_info['size'],
            "last_modified": file_info['last_modified'],
            "etag": file_info['etag']
        }
        
        source_id = get_or_create_source(
            kind="s3",
            uri=source_uri,
            bucket=bucket_name,
            s3_key=file_key,
            s3_etag=file_info['etag'],
            last_modified=file_info['last_modified'],
            metadata=source_metadata
        )
        

        doc_metadata = {
            "file_size": file_info['size'],
            "last_modified": file_info['last_modified'],
            "processing_timestamp": datetime.now().isoformat()
        }
        
        document_id = create_document(
            source_id=source_id,
            ingestion_run_id=ingestion_run_id,
            external_id=file_key,
            uri=source_uri,
            mime_type="application/pdf",
            title=os.path.basename(file_key),
            content=text,
            metadata=doc_metadata
        )
        

        chunks = chunk_text(text)
        print(f"Created {len(chunks)} chunks")
        
        try:
            chunk_ids = create_chunks(document_id, chunks)
            print(f"Successfully created {len(chunk_ids)} chunk records in database")
        except Exception as e:
            print(f"ERROR creating chunks: {str(e)}")
            raise
        

        embeddings = embed_chunks(chunks)
        print(f"Generated {len(embeddings)} embeddings")
        

        try:
            embedding_ids = create_chunk_embeddings(chunk_ids, embeddings, model_id)
            print(f"Successfully stored {len(embedding_ids)} embeddings via API")
        except Exception as e:
            print(f"ERROR storing embeddings: {str(e)}")
            raise
        
        return True
        
    except Exception as e:
        print(f"Error processing {file_key}: {str(e)}")
        return False

def main():
    try:

        bucket_name = os.getenv('AWS_S3_BUCKET')
        if not bucket_name:
            print("Please set AWS_S3_BUCKET in your .env file")
            return
        

        try:
            with get_api_client() as client:
                response = client.get("/health")
                response.raise_for_status()
                print("Connected to API successfully")
        except Exception as e:
            print(f"Failed to connect to API: {str(e)}")
            print("Make sure the API service is running on the configured URL")
            return
        
        try:
    
            model_id = get_or_create_embedding_model()
            print(f"Using embedding model ID: {model_id}")
            
    
            ingestion_run_id = create_ingestion_run(
                source_label=f"S3 bucket: {bucket_name}",
                notes="Automated PDF ingestion from S3"
            )
            print(f"Started ingestion run: {ingestion_run_id}")
            
    
            update_ingestion_run_status(ingestion_run_id, "running")
            
    
            print("Fetching files from S3...")
            files = list_s3_files()
            
            if not files:
                print("No files found in the bucket")
                update_ingestion_run_status(ingestion_run_id, "succeeded", "No files to process")
                return
            
    
            pdf_files = [f for f in files if f['key'].lower().endswith('.pdf')]
            print(f"Found {len(pdf_files)} PDF files to process")
            
            successful_files = 0
            failed_files = 0
            
            for file_info in pdf_files:
        
                precheck = precheck_s3_file(bucket_name, file_info)
                
                if not precheck.get("should_download", True):
                    print(f"SKIP download (etag match): {file_info['key']}")
            
                    continue
                
                success = process_s3_file_to_api(
                    bucket_name, file_info, ingestion_run_id, model_id
                )
                
                if success:
                    successful_files += 1
                else:
                    failed_files += 1
                
                print("-" * 50)
            
    
            if failed_files == 0:
                status = "succeeded"
                notes = f"Successfully processed {successful_files} files"
            elif successful_files == 0:
                status = "failed"
                notes = f"Failed to process all {failed_files} files"
            else:
                status = "succeeded"
                notes = f"Processed {successful_files} files successfully, {failed_files} failed"
            
            update_ingestion_run_status(ingestion_run_id, status, notes)
            print(f"\nIngestion completed: {notes}")
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
    
            try:
                if 'ingestion_run_id' in locals():
                    update_ingestion_run_status(ingestion_run_id, "failed", str(e))
            except:
                pass
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()