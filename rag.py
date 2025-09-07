import boto3
import os
import io
from dotenv import load_dotenv
from typing import List, Dict
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings

load_dotenv()

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


def main():
    try:
        bucket_name = os.getenv('AWS_S3_BUCKET')
        if not bucket_name:
            print("Please set AWS_S3_BUCKET in your .env file")
            return
        
        print("Processing PDF files in S3 bucket:")
        files = list_s3_files()
        
        if files:
            for file in files:
                if file['key'].lower().endswith('.pdf'):
                    print(f"\nPDF File: {file['key']}")
                    print(f"Size: {file['size']} bytes")
                    
                    try:
                        text = get_pdf_text_from_s3(bucket_name, file['key'])
                        print(f"Extracted text length: {len(text)} characters")
                        
                        chunks = chunk_text(text)
                        print(f"Created {len(chunks)} chunks")
                        
                        embeddings = embed_chunks(chunks)
                        print(f"Generated {len(embeddings)} embeddings")
                        print(f"Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
                        
                        for i, chunk in enumerate(chunks[:3]):
                            print(f"\nChunk {i+1} (first 150 chars): {chunk[:150]}...")
                            print(f"Embedding vector length: {len(embeddings[i])}")
                        
                        if len(chunks) > 3:
                            print(f"... and {len(chunks) - 3} more chunks with embeddings")
                            
                    except Exception as e:
                        print(f"Error processing file: {str(e)}")
                    
                    print("-" * 50)
        else:
            print("No files found in the bucket")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()