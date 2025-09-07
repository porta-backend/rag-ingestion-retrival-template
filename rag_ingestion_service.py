"""
FastAPI service for RAG database operations.
Handles all database interactions asynchronously.
"""

import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, status, Query
from pydantic import BaseModel, field_validator
import asyncpg
from pgvector.asyncpg import register_vector

load_dotenv()

app = FastAPI(title="RAG Database API", version="1.0.0")

# Database connection pool
db_pool = None

async def get_db_pool():
    """Get database connection pool"""
    global db_pool
    if db_pool is None:
        db_pool = await asyncpg.create_pool(
            host=os.getenv('HOST'),
            port=int(os.getenv('PORT', 5432)),
            user=os.getenv('USERNAME'),
            password=os.getenv('PASSWORD'),
            database=os.getenv('DATABASE'),
            min_size=1,
            max_size=10
        )
        # Register vector extension for all connections
        async with db_pool.acquire() as conn:
            await register_vector(conn)
    return db_pool

@app.on_event("startup")
async def startup_event():
    """Initialize database pool on startup"""
    await get_db_pool()

@app.on_event("shutdown")
async def shutdown_event():
    """Close database pool on shutdown"""
    global db_pool
    if db_pool:
        await db_pool.close()

# Pydantic models for request/response
class EmbeddingModelCreate(BaseModel):
    name: str = "amazon.titan-embed-text-v1"
    provider: str = "bedrock"
    embedding_dim: int = 1536
    distance_metric: str = "cosine"
    version: str = "v1"

class EmbeddingModelResponse(BaseModel):
    id: str
    name: str
    provider: str
    embedding_dim: int
    distance_metric: str
    version: str


    @field_validator('id', mode='before')
    @classmethod
    def convert_uuid_to_str(cls, v):
        if hasattr(v, '__str__'):
            return str(v)
        return v
class IngestionRunCreate(BaseModel):
    source_label: Optional[str] = None
    notes: Optional[str] = None

class IngestionRunResponse(BaseModel):
    id: str
    status: str
    source_label: Optional[str]
    notes: Optional[str]


    @field_validator('id', mode='before')
    @classmethod
    def convert_uuid_to_str(cls, v):
        if hasattr(v, '__str__'):
            return str(v)
        return v
class IngestionRunUpdate(BaseModel):
    status: str
    notes: Optional[str] = None

class SourceCreate(BaseModel):
    kind: str
    uri: str
    bucket: Optional[str] = None
    s3_key: Optional[str] = None
    s3_etag: Optional[str] = None
    content_hash: Optional[str] = None
    metadata: Optional[Dict] = None

class SourceResponse(BaseModel):
    id: str
    kind: str
    uri: str
    bucket: Optional[str]
    s3_key: Optional[str]
    s3_etag: Optional[str]
    content_hash: Optional[str]
    metadata: Optional[Dict]


    @field_validator('id', mode='before')
    @classmethod
    def convert_uuid_to_str(cls, v):
        if hasattr(v, '__str__'):
            return str(v)
        return v
class DocumentCreate(BaseModel):
    source_id: str
    ingestion_run_id: str
    external_id: Optional[str] = None
    uri: Optional[str] = None
    mime_type: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    language: Optional[str] = None
    content: Optional[str] = None
    custom_tags: Optional[List[str]] = None
    metadata: Optional[Dict] = None

class DocumentResponse(BaseModel):
    id: str
    source_id: str
    ingestion_run_id: str
    external_id: Optional[str]
    uri: Optional[str]
    mime_type: Optional[str]
    title: Optional[str]

    @field_validator('id', 'source_id', 'ingestion_run_id', mode='before')
    @classmethod
    def convert_uuid_to_str(cls, v):
        if hasattr(v, '__str__'):
            return str(v)
        return v
class ChunkCreate(BaseModel):
    document_id: str
    chunks: List[str]
    metadata_list: Optional[List[Dict]] = None

class ChunkResponse(BaseModel):
    chunk_ids: List[str]


    @field_validator('chunk_ids', mode='before')
    @classmethod
    def convert_uuid_list_to_str(cls, v):
        if isinstance(v, list):
            return [str(item) if hasattr(item, '__str__') else item for item in v]
        return v
class EmbeddingCreate(BaseModel):
    chunk_ids: List[str]
    embeddings: List[List[float]]
    model_id: str

class EmbeddingResponse(BaseModel):
    embedding_ids: List[str]

    @field_validator('embedding_ids', mode='before')
    @classmethod
    def convert_uuid_list_to_str(cls, v):
        if isinstance(v, list):
            return [str(item) if hasattr(item, '__str__') else item for item in v]
        return v

# New models for incremental processing
class S3Precheck(BaseModel):
    bucket: str
    key: str
    etag: Optional[str] = None
    size: Optional[int] = None
    last_modified: Optional[str] = None  # ISO format

class S3PrecheckResp(BaseModel):
    should_download: bool
    source_id: Optional[str] = None

class ChecksumExistsResp(BaseModel):
    exists: bool = False
    document_id: Optional[str] = None

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/precheck/s3", response_model=S3PrecheckResp)
async def precheck_s3(obj: S3Precheck):
    """Check if S3 object should be downloaded based on fingerprint comparison"""
    pool = await get_db_pool()
    uri = f"s3://{obj.bucket}/{obj.key}"
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
          SELECT id, s3_etag, metadata
          FROM sources
          WHERE kind='s3' AND uri=$1
        """, uri)

        if not row:
            return S3PrecheckResp(should_download=True, source_id=None)

        same_etag = (obj.etag is not None and obj.etag == row["s3_etag"])
        # Optionally also compare size/last_modified from metadata:
        # stored = json.loads(row["metadata"]) if row["metadata"] else {}
        # same_size = stored.get("size") == obj.size
        # same_lm = stored.get("last_modified") == obj.last_modified
        
        # Simple, conservative check: if ETag matches, skip download
        if same_etag:
            # Refresh last_seen_at for housekeeping
            await conn.execute("UPDATE sources SET last_seen_at = now() WHERE id=$1", row["id"])
            return S3PrecheckResp(should_download=False, source_id=str(row["id"]))

        return S3PrecheckResp(should_download=True, source_id=str(row["id"]))

@app.get("/documents/checksum-exists", response_model=ChecksumExistsResp)
async def checksum_exists(sha: str = Query(..., min_length=64, max_length=64)):
    """Check if a document with the given SHA-256 checksum already exists"""
    pool = await get_db_pool()
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
          SELECT id FROM documents WHERE checksum_sha256=$1
        """, sha)
        
        if row:
            return ChecksumExistsResp(exists=True, document_id=str(row["id"]))
        return ChecksumExistsResp(exists=False)

@app.post("/embedding-models", response_model=EmbeddingModelResponse)
async def create_or_get_embedding_model(model: EmbeddingModelCreate):
    """Create or get embedding model"""
    pool = await get_db_pool()
    
    async with pool.acquire() as conn:
        # Check if model exists
        existing = await conn.fetchrow("""
            SELECT id, name, provider, embedding_dim, distance_metric, version 
            FROM embedding_models 
            WHERE name = $1 AND provider = $2 AND version = $3
        """, model.name, model.provider, model.version)
        
        if existing:
            return EmbeddingModelResponse(**dict(existing))
        
        # Create new model (let database generate UUID)
        row = await conn.fetchrow("""
            INSERT INTO embedding_models (name, provider, embedding_dim, distance_metric, version)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id, name, provider, embedding_dim, distance_metric, version
        """, model.name, model.provider, model.embedding_dim, model.distance_metric, model.version)
        
        return EmbeddingModelResponse(**dict(row))

@app.post("/ingestion-runs", response_model=IngestionRunResponse)
async def create_ingestion_run(run: IngestionRunCreate):
    """Create a new ingestion run"""
    pool = await get_db_pool()
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            INSERT INTO ingestion_runs (source_label, notes)
            VALUES ($1, $2)
            RETURNING id, status, source_label, notes
        """, run.source_label, run.notes)
        
        return IngestionRunResponse(**dict(row))

@app.put("/ingestion-runs/{run_id}")
async def update_ingestion_run(run_id: str, update: IngestionRunUpdate):
    """Update ingestion run status"""
    pool = await get_db_pool()
    
    async with pool.acquire() as conn:
        if update.status in ['succeeded', 'failed']:
            await conn.execute("""
                UPDATE ingestion_runs 
                SET status = $1, finished_at = now(), notes = COALESCE($2, notes)
                WHERE id = $3
            """, update.status, update.notes, run_id)
        else:
            await conn.execute("""
                UPDATE ingestion_runs 
                SET status = $1, notes = COALESCE($2, notes)
                WHERE id = $3
            """, update.status, update.notes, run_id)
        
        return {"message": "Ingestion run updated successfully"}

@app.post("/sources", response_model=SourceResponse)
async def create_or_get_source(source: SourceCreate):
    """Create or get source using UPSERT for idempotent operations"""
    pool = await get_db_pool()
    m = source.metadata or {}
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
          INSERT INTO sources (kind, uri, bucket, s3_key, s3_etag, content_hash, metadata, first_seen_at, last_seen_at)
          VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb, now(), now())
          ON CONFLICT (kind, uri)
          DO UPDATE SET s3_etag=EXCLUDED.s3_etag,
                        metadata=EXCLUDED.metadata,
                        last_seen_at=now()
          RETURNING id, kind, uri, bucket, s3_key, s3_etag, content_hash, metadata
        """, source.kind, source.uri, source.bucket, source.s3_key,
             source.s3_etag, source.content_hash, json.dumps(m))
        
        rec = dict(row)
        # asyncpg returns text for jsonb unless codecs are set; make sure it's dict:
        if isinstance(rec.get("metadata"), str):
            rec["metadata"] = json.loads(rec["metadata"])
        return SourceResponse(**rec)

@app.post("/documents", response_model=DocumentResponse)
async def create_document(document: DocumentCreate):
    """Create document using UPSERT for idempotent operations"""
    pool = await get_db_pool()
    m = document.metadata or {}
    content_hash = hashlib.sha256(document.content.encode("utf-8")).hexdigest() if document.content else None
    token_count = len(document.content) // 4 if document.content else None

    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
          INSERT INTO documents (source_id, ingestion_run_id, external_id, uri, mime_type, title, author, language,
                                 checksum_sha256, token_count, custom_tags, metadata, created_at, updated_at)
          VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12::jsonb, now(), now())
          ON CONFLICT (checksum_sha256)
          DO UPDATE SET updated_at=now(), state='active'
          RETURNING id, source_id, ingestion_run_id, external_id, uri, mime_type, title
        """, document.source_id, document.ingestion_run_id, document.external_id,
             document.uri, document.mime_type, document.title, document.author, document.language,
             content_hash, token_count, document.custom_tags, json.dumps(m))
        return DocumentResponse(**dict(row))

@app.post("/chunks", response_model=ChunkResponse)
async def create_chunks(chunk_data: ChunkCreate):
    """Create chunks for a document using bulk operations"""
    print(f"[CHUNKS API] Received request: {len(chunk_data.chunks)} chunks for document {chunk_data.document_id}")
    
    pool = await get_db_pool()
    metas = chunk_data.metadata_list or [{} for _ in chunk_data.chunks]
    n = len(chunk_data.chunks)
    if len(metas) != n:
        raise HTTPException(status_code=400, detail="metadata_list length must match chunks")

    idxs = list(range(n))
    tokens = [len(c)//4 for c in chunk_data.chunks]
    metas_json = [json.dumps(m) for m in metas]

    print(f"[CHUNKS API] Preparing to insert {n} chunks (letting database generate UUIDs)")

    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                print(f"[CHUNKS API] Starting transaction for document {chunk_data.document_id}")
                
                delete_result = await conn.execute("DELETE FROM chunks WHERE document_id=$1", chunk_data.document_id)
                print(f"[CHUNKS API] Deleted old chunks: {delete_result}")
                
                # Let database generate UUIDs and return them
                rows = await conn.fetch("""
                  INSERT INTO chunks (document_id, chunk_index, content, token_count, metadata)
                  SELECT $1::uuid, UNNEST($2::int[]), UNNEST($3::text[]), UNNEST($4::int[]), UNNEST($5::jsonb[])
                  RETURNING id
                """, chunk_data.document_id, idxs, chunk_data.chunks, tokens, metas_json)
                
                ids = [str(row['id']) for row in rows]
                print(f"[CHUNKS API] Successfully inserted {len(ids)} chunks with database-generated UUIDs")
                print(f"[CHUNKS API] Generated chunk IDs, first 3: {ids[:3]}")
                
                # Verify chunks were inserted
                count = await conn.fetchval("SELECT COUNT(*) FROM chunks WHERE document_id=$1", chunk_data.document_id)
                print(f"[CHUNKS API] Verification: {count} chunks found in database for document {chunk_data.document_id}")
                
    except Exception as e:
        print(f"[CHUNKS API] ERROR during chunk creation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    return ChunkResponse(chunk_ids=ids)

@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(embedding_data: EmbeddingCreate):
    """Create chunk embeddings with idempotent operations"""
    print(f"[EMBEDDING API] Received request: {len(embedding_data.chunk_ids)} chunks, model_id: {embedding_data.model_id}")
    
    pool = await get_db_pool()
    if len(embedding_data.chunk_ids) != len(embedding_data.embeddings):
        raise HTTPException(status_code=400, detail="chunk_ids and embeddings length must match")

    embedding_ids = []
    
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                print(f"[EMBEDDING API] Starting transaction for {len(embedding_data.chunk_ids)} embeddings")
                
                # First, verify all chunk IDs exist
                existing_chunks = await conn.fetch("""
                  SELECT id FROM chunks WHERE id = ANY($1::uuid[])
                """, embedding_data.chunk_ids)
                existing_chunk_ids = [str(row['id']) for row in existing_chunks]
                print(f"[EMBEDDING API] Found {len(existing_chunk_ids)} existing chunks out of {len(embedding_data.chunk_ids)} requested")
                
                if len(existing_chunk_ids) != len(embedding_data.chunk_ids):
                    missing_chunks = set(embedding_data.chunk_ids) - set(existing_chunk_ids)
                    print(f"[EMBEDDING API] ERROR: Missing chunks: {list(missing_chunks)[:5]}...")
                    raise HTTPException(status_code=400, detail=f"Missing {len(missing_chunks)} chunk(s) in database")
                
                # Mark old as not current (idempotent upsert semantics)
                result = await conn.execute("""
                  UPDATE chunk_embeddings
                  SET is_current = FALSE, updated_at = now()
                  WHERE chunk_id = ANY($1::uuid[]) AND model_id = $2 AND is_current = TRUE
                """, embedding_data.chunk_ids, embedding_data.model_id)
                print(f"[EMBEDDING API] Updated {result.split(' ')[-1] if result else '0'} old embeddings to not current")

                # Insert new embeddings individually (pgvector works better this way)
                for i, (chunk_id, embedding_vector) in enumerate(zip(embedding_data.chunk_ids, embedding_data.embeddings)):
                    # With pgvector.asyncpg registered, we can pass the list directly
                    row = await conn.fetchrow("""
                      INSERT INTO chunk_embeddings (chunk_id, model_id, embedding, is_current, created_at, updated_at)
                      VALUES ($1, $2, $3, TRUE, now(), now())
                      RETURNING id
                    """, chunk_id, embedding_data.model_id, embedding_vector)
                    embedding_ids.append(str(row['id']))
                
                print(f"[EMBEDDING API] Successfully inserted {len(embedding_ids)} embeddings")
    
    except Exception as e:
        print(f"[EMBEDDING API] ERROR during insertion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    return EmbeddingResponse(embedding_ids=embedding_ids)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
