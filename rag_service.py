
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
from langchain_aws import BedrockEmbeddings

load_dotenv()

app = FastAPI(title="RAG Database API", version="1.0.0")

db_pool = None

async def get_db_pool():
    global db_pool
    if db_pool is None:
        db_pool = await asyncpg.create_pool(
            host=os.getenv('HOST'),
            port=int(os.getenv('PORT', 5432)),
            user=os.getenv('USERNAME'),
            password=os.getenv('PASSWORD'),
            database=os.getenv('DATABASE'),
            min_size=1,
            max_size=10,
            init=register_vector
        )
    return db_pool

@app.on_event("startup")
async def startup_event():
    await get_db_pool()

@app.on_event("shutdown")
async def shutdown_event():
    global db_pool
    if db_pool:
        await db_pool.close()

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
    last_modified: Optional[str] = None  # ISO format timestamp
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

class S3Precheck(BaseModel):
    bucket: str
    key: str
    etag: Optional[str] = None
    size: Optional[int] = None
    last_modified: Optional[str] = None

class S3PrecheckResp(BaseModel):
    should_download: bool
    source_id: Optional[str] = None

class ChecksumExistsResp(BaseModel):
    exists: bool = False
    document_id: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    similarity_threshold: float = 0.7
    max_results: int = 10
    model_name: str = "amazon.titan-embed-text-v1"

class ChunkResult(BaseModel):
    chunk_id: str
    content: str
    similarity_score: float
    document_title: Optional[str] = None
    document_author: Optional[str] = None
    source_uri: Optional[str] = None
    chunk_index: Optional[int] = None
    token_count: Optional[int] = None
    metadata: Optional[Dict] = None

class QueryResponse(BaseModel):
    query: str
    results: List[ChunkResult]
    total_results: int
    similarity_threshold: float
    model_name: str

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/precheck/s3", response_model=S3PrecheckResp)
async def precheck_s3(obj: S3Precheck):
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
        
        stored_metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        stored_last_modified = stored_metadata.get("last_modified")
        same_last_modified = (obj.last_modified is not None and 
                             stored_last_modified is not None and 
                             obj.last_modified == stored_last_modified)
        if same_etag and same_last_modified:
            await conn.execute("UPDATE sources SET last_seen_at = now() WHERE id=$1", row["id"])
            return S3PrecheckResp(should_download=False, source_id=str(row["id"]))

        return S3PrecheckResp(should_download=True, source_id=str(row["id"]))

@app.get("/documents/checksum-exists", response_model=ChecksumExistsResp)
async def checksum_exists(sha: str = Query(..., min_length=64, max_length=64)):
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
    pool = await get_db_pool()
    
    async with pool.acquire() as conn:
        existing = await conn.fetchrow("""
            SELECT id, name, provider, embedding_dim, distance_metric, version 
            FROM embedding_models 
            WHERE name = $1 AND provider = $2 AND version = $3
        """, model.name, model.provider, model.version)
        
        if existing:
            return EmbeddingModelResponse(**dict(existing))
        
        row = await conn.fetchrow("""
            INSERT INTO embedding_models (name, provider, embedding_dim, distance_metric, version)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id, name, provider, embedding_dim, distance_metric, version
        """, model.name, model.provider, model.embedding_dim, model.distance_metric, model.version)
        
        return EmbeddingModelResponse(**dict(row))

@app.post("/ingestion-runs", response_model=IngestionRunResponse)
async def create_ingestion_run(run: IngestionRunCreate):
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
    pool = await get_db_pool()
    m = source.metadata or {}
    
    if source.kind == 's3' and source.last_modified:
        m['last_modified'] = source.last_modified
    
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
        if isinstance(rec.get("metadata"), str):
            rec["metadata"] = json.loads(rec["metadata"])
        return SourceResponse(**rec)

@app.post("/documents", response_model=DocumentResponse)
async def create_document(document: DocumentCreate):
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
    print(f"[CHUNKS API] Received request: {len(chunk_data.chunks)} chunks for document {chunk_data.document_id}")
    
    pool = await get_db_pool()
    metas = chunk_data.metadata_list or [{} for _ in chunk_data.chunks]
    n = len(chunk_data.chunks)
    if len(metas) != n:
        raise HTTPException(status_code=400, detail="metadata_list length must match chunks")

    idxs = list(range(n))
    tokens = [len(c)//4 for c in chunk_data.chunks]
    metas_json = [json.dumps(m) for m in metas]

    print(f"[CHUNKS API] Preparing to insert {n} chunks")

    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                print(f"[CHUNKS API] Starting transaction for document {chunk_data.document_id}")
                
                delete_result = await conn.execute("DELETE FROM chunks WHERE document_id=$1", chunk_data.document_id)
                print(f"[CHUNKS API] Deleted old chunks: {delete_result}")
                
                rows = await conn.fetch("""
                  INSERT INTO chunks (document_id, chunk_index, content, token_count, metadata)
                  SELECT $1::uuid, UNNEST($2::int[]), UNNEST($3::text[]), UNNEST($4::int[]), UNNEST($5::jsonb[])
                  RETURNING id
                """, chunk_data.document_id, idxs, chunk_data.chunks, tokens, metas_json)
                
                ids = [str(row['id']) for row in rows]
                print(f"[CHUNKS API] Successfully inserted {len(ids)} chunks")
                print(f"[CHUNKS API] Generated chunk IDs, first 3: {ids[:3]}")
                
                count = await conn.fetchval("SELECT COUNT(*) FROM chunks WHERE document_id=$1", chunk_data.document_id)
                print(f"[CHUNKS API] Verification: {count} chunks found in database for document {chunk_data.document_id}")
                
    except Exception as e:
        print(f"[CHUNKS API] ERROR during chunk creation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    return ChunkResponse(chunk_ids=ids)

@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(embedding_data: EmbeddingCreate):
    print(f"[EMBEDDING API] Received request: {len(embedding_data.chunk_ids)} chunks, model_id: {embedding_data.model_id}")
    
    pool = await get_db_pool()
    if len(embedding_data.chunk_ids) != len(embedding_data.embeddings):
        raise HTTPException(status_code=400, detail="chunk_ids and embeddings length must match")
    
    for i, embedding in enumerate(embedding_data.embeddings):
        if not isinstance(embedding, list):
            raise HTTPException(status_code=400, detail=f"Embedding {i} is not a list, got {type(embedding)}")
        if not all(isinstance(x, (int, float)) for x in embedding):
            raise HTTPException(status_code=400, detail=f"Embedding {i} contains non-numeric values")
        if len(embedding) == 0:
            raise HTTPException(status_code=400, detail=f"Embedding {i} is empty")
    
    print(f"[EMBEDDING API] Validated {len(embedding_data.embeddings)} embeddings, first embedding length: {len(embedding_data.embeddings[0]) if embedding_data.embeddings else 'N/A'}")

    embedding_ids = []
    
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                print(f"[EMBEDDING API] Starting transaction for {len(embedding_data.chunk_ids)} embeddings")
                
                existing_chunks = await conn.fetch("""
                  SELECT id FROM chunks WHERE id = ANY($1::uuid[])
                """, embedding_data.chunk_ids)
                existing_chunk_ids = [str(row['id']) for row in existing_chunks]
                print(f"[EMBEDDING API] Found {len(existing_chunk_ids)} existing chunks out of {len(embedding_data.chunk_ids)} requested")
                
                if len(existing_chunk_ids) != len(embedding_data.chunk_ids):
                    missing_chunks = set(embedding_data.chunk_ids) - set(existing_chunk_ids)
                    print(f"[EMBEDDING API] ERROR: Missing chunks: {list(missing_chunks)[:5]}...")
                    raise HTTPException(status_code=400, detail=f"Missing {len(missing_chunks)} chunk(s) in database")
                
                result = await conn.execute("""
                  UPDATE chunk_embeddings
                  SET is_current = FALSE, updated_at = now()
                  WHERE chunk_id = ANY($1::uuid[]) AND model_id = $2 AND is_current = TRUE
                """, embedding_data.chunk_ids, embedding_data.model_id)
                print(f"[EMBEDDING API] Updated {result.split(' ')[-1] if result else '0'} old embeddings to not current")

                for i, (chunk_id, embedding_vector) in enumerate(zip(embedding_data.chunk_ids, embedding_data.embeddings)):
                    try:
                        print(f"[EMBEDDING API] Inserting embedding {i+1}/{len(embedding_data.embeddings)}, vector length: {len(embedding_vector)}")
                        row = await conn.fetchrow("""
                          INSERT INTO chunk_embeddings (chunk_id, model_id, embedding, is_current, created_at, updated_at)
                          VALUES ($1, $2, $3, TRUE, now(), now())
                          RETURNING id
                        """, chunk_id, embedding_data.model_id, embedding_vector)
                        embedding_ids.append(str(row['id']))
                    except Exception as e:
                        print(f"[EMBEDDING API] ERROR inserting embedding {i+1}: {str(e)}")
                        print(f"[EMBEDDING API] Chunk ID: {chunk_id}, Model ID: {embedding_data.model_id}")
                        print(f"[EMBEDDING API] Vector type: {type(embedding_vector)}, length: {len(embedding_vector) if hasattr(embedding_vector, '__len__') else 'N/A'}")
                        raise
                
                print(f"[EMBEDDING API] Successfully inserted {len(embedding_ids)} embeddings")
    
    except Exception as e:
        print(f"[EMBEDDING API] ERROR during insertion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    return EmbeddingResponse(embedding_ids=embedding_ids)

@app.get("/documents")
async def get_documents(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
    pool = await get_db_pool()
    
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT d.id, d.title, d.author, d.uri, d.mime_type, d.created_at,
                   s.kind as source_kind, s.uri as source_uri
            FROM documents d
            JOIN sources s ON d.source_id = s.id
            WHERE d.state = 'active'
            ORDER BY d.created_at DESC
            LIMIT $1 OFFSET $2
        """, limit, offset)
        
        total_count = await conn.fetchval("SELECT COUNT(*) FROM documents WHERE state = 'active'")
        
        return {
            "documents": [dict(row) for row in rows],
            "total_count": total_count,
            "limit": limit,
            "offset": offset
        }

@app.get("/documents/{document_id}/chunks")
async def get_document_chunks(document_id: str, limit: int = Query(100, ge=1, le=1000)):
    pool = await get_db_pool()
    
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, chunk_index, content, token_count, metadata, created_at
            FROM chunks
            WHERE document_id = $1
            ORDER BY chunk_index ASC
            LIMIT $2
        """, document_id, limit)
        
        return {
            "document_id": document_id,
            "chunks": [dict(row) for row in rows],
            "count": len(rows)
        }

@app.post("/search/similar")
async def search_similar_chunks(
    query_embedding: List[float],
    similarity_threshold: float = Query(0.7, ge=0.0, le=1.0),
    max_results: int = Query(10, ge=1, le=100),
    model_name: str = Query("amazon.titan-embed-text-v1")
):
    pool = await get_db_pool()
    
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT * FROM find_similar_chunks($1::vector(1536), $2::float, $3::int, $4::text)
        """, query_embedding, similarity_threshold, max_results, model_name)
        
        return {
            "query_embedding_dim": len(query_embedding),
            "similarity_threshold": similarity_threshold,
            "max_results": max_results,
            "model_name": model_name,
            "results": [dict(row) for row in rows],
            "count": len(rows)
        }

@app.get("/chunks/{chunk_id}")
async def get_chunk_details(chunk_id: str):
    pool = await get_db_pool()
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT c.id, c.chunk_index, c.content, c.token_count, c.metadata, c.created_at,
                   d.title as document_title, d.author as document_author,
                   s.kind as source_kind, s.uri as source_uri
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            JOIN sources s ON d.source_id = s.id
            WHERE c.id = $1
        """, chunk_id)
        
        if not row:
            raise HTTPException(status_code=404, detail="Chunk not found")
        
        return dict(row)

@app.get("/embedding-models")
async def get_embedding_models():
    pool = await get_db_pool()
    
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, name, provider, embedding_dim, distance_metric, version, created_at
            FROM embedding_models
            ORDER BY created_at DESC
        """)
        
        return {
            "models": [dict(row) for row in rows],
            "count": len(rows)
        }

@app.get("/debug/stats")
async def get_debug_stats():
    pool = await get_db_pool()
    
    async with pool.acquire() as conn:
        embedding_count = await conn.fetchval("SELECT COUNT(*) FROM chunk_embeddings WHERE is_current = TRUE")
        chunk_count = await conn.fetchval("SELECT COUNT(*) FROM chunks")
        document_count = await conn.fetchval("SELECT COUNT(*) FROM documents WHERE state = 'active'")
        model_count = await conn.fetchval("SELECT COUNT(*) FROM embedding_models")
        
        sample_chunk = await conn.fetchrow("""
            SELECT 
                c.id,
                c.content,
                c.chunk_index,
                d.title,
                em.name as model_name,
                ce.embedding
            FROM chunks c
            JOIN chunk_embeddings ce ON c.id = ce.chunk_id
            JOIN documents d ON c.document_id = d.id
            JOIN embedding_models em ON ce.model_id = em.id
            WHERE ce.is_current = TRUE
            LIMIT 1
        """)
        
        return {
            "stats": {
                "total_embeddings": embedding_count,
                "total_chunks": chunk_count,
                "total_documents": document_count,
                "total_models": model_count
            },
            "sample_chunk": {
                "id": str(sample_chunk["id"]) if sample_chunk else None,
                "content_preview": sample_chunk["content"][:100] + "..." if sample_chunk else None,
                "chunk_index": sample_chunk["chunk_index"] if sample_chunk else None,
                "document_title": sample_chunk["title"] if sample_chunk else None,
                "model_name": sample_chunk["model_name"] if sample_chunk else None,
                "embedding_length": len(sample_chunk["embedding"]) if sample_chunk else None
            } if sample_chunk else None
        }

@app.post("/query", response_model=QueryResponse)
async def query_chunks(request: QueryRequest):
    try:
        embeddings = BedrockEmbeddings(
            model_id=request.model_name,
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        
        query_embedding = embeddings.embed_query(request.query)
        
        pool = await get_db_pool()
        
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM find_similar_chunks($1::vector(1536), $2::float, $3::int, $4::text)
            """, query_embedding, request.similarity_threshold, request.max_results, request.model_name)
            
            results = []
            for row in rows:
                chunk_result = ChunkResult(
                    chunk_id=str(row['chunk_id']),
                    content=row['content'],
                    similarity_score=float(row['similarity']),
                    document_title=row.get('document_title'),
                    document_author=row.get('document_author'),
                    source_uri=row.get('source_uri'),
                    chunk_index=row.get('chunk_index'),
                    token_count=row.get('token_count'),
                    metadata=row.get('metadata')
                )
                results.append(chunk_result)
            
            return QueryResponse(
                query=request.query,
                results=results,
                total_results=len(results),
                similarity_threshold=request.similarity_threshold,
                model_name=request.model_name
            )
            
    except Exception as e:
        print(f"Error in query endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing query: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
