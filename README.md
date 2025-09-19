<<<<<<< HEAD
# RAG Ingestion & Retrieval Template

A comprehensive Python-based RAG (Retrieval-Augmented Generation) system template that processes PDF documents from AWS S3, extracts text, creates embeddings using OpenAI's text-embedding-ada-002 model, and stores everything in a PostgreSQL database with pgvector support for efficient similarity search.

## 🚀 Features

- **S3 Integration**: Automatically downloads and processes PDF files from AWS S3 buckets
- **PDF Text Extraction**: Extracts text content from PDF documents using PyPDF2
- **Text Chunking**: Splits large documents into manageable chunks with configurable overlap
- **OpenAI Embeddings**: Uses OpenAI text-embedding-ada-002 model to generate vector representations
- **PostgreSQL + pgvector Storage**: Stores documents, chunks, and embeddings in a structured database
- **Vector Similarity Search**: Efficient similarity search using pgvector's HNSW indexing
- **Ingestion Tracking**: Tracks ingestion runs with status and metadata
- **Comprehensive Metadata**: Stores rich metadata for sources, documents, and chunks
- **Error Handling**: Comprehensive error handling for robust operation
- **Conversational RAG**: Chat interface with session management and conversation history
- **FastAPI Services**: RESTful APIs for both ingestion and retrieval operations

## 📋 Prerequisites

- Python 3.7+
- OpenAI API key
- AWS S3 bucket with PDF files
- AWS credentials configured
- PostgreSQL database with pgvector extension
- Database credentials configured

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/porta-backend/rag-ingestion-retrival-template.git
cd rag-ingestion-retrival-template
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

1. Create a `.env` file with your configuration:
```env
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=us-east-1
AWS_S3_BUCKET=your-bucket-name

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration (for API service)
HOST=your-database-host
PORT=5432
USERNAME=your-database-username
PASSWORD=your-database-password
DATABASE=your-database-name

# API Configuration (for RAG client)
API_BASE_URL=http://localhost:8000
```

## 🗄️ Database Setup

Before running the system, ensure your PostgreSQL database has the required schema. The system expects the following tables:

- `embedding_models`: Stores embedding model configurations
- `ingestion_runs`: Tracks ingestion process runs
- `sources`: Stores source information (S3 buckets, files, etc.)
- `documents`: Stores document metadata and content
- `chunks`: Stores text chunks with full-text search support
- `chunk_embeddings`: Stores vector embeddings with pgvector support

See the `schema_rag_ingestion.sql` file for the complete table definitions.

## 🚀 Usage

### 1. Start the Database API Service

First, start the database API service:

```bash
python rag_service.py
```

This will start the FastAPI service on `http://localhost:8000`. You can view the API documentation at `http://localhost:8000/docs`.

### 2. Run the Main Ingestion Process

In a new terminal, run the RAG ingestion:

```bash
python rag_ingestion.py
```

The script will:
1. Connect to the API service and S3 bucket
2. Create or get the embedding model configuration via API
3. Start an ingestion run via API
4. List all PDF files in the S3 bucket
5. For each PDF file:
   - Download and extract text
   - Create document and source records via API
   - Split text into chunks
   - Generate embeddings using OpenAI text-embedding-ada-002 model
   - Store everything in the database via API
6. Update ingestion run status via API

### 3. Start the Conversational RAG Service

In another terminal, start the conversational RAG service:

```bash
python rag_retrival.py
```

This will start the conversational RAG API on `http://localhost:8001`.

## 📡 API Endpoints

### Database API (Port 8000)

- `GET /health` - Health check
- `POST /embedding-models` - Create/get embedding models
- `POST /ingestion-runs` - Create ingestion runs
- `PUT /ingestion-runs/{run_id}` - Update ingestion run status
- `POST /sources` - Create/get sources
- `POST /documents` - Create documents
- `POST /chunks` - Create chunks
- `POST /embeddings` - Create embeddings
- `POST /query` - Query for relevant chunks using vector similarity
- `GET /documents` - Get documents with pagination
- `GET /documents/{document_id}/chunks` - Get chunks for a specific document
- `GET /chunks/{chunk_id}` - Get detailed information about a specific chunk
- `GET /embedding-models` - Get all embedding models
- `GET /debug/stats` - Get debug statistics about the database

### Conversational RAG API (Port 8001)

- `GET /health` - Health check
- `POST /chat` - Chat with the RAG system
- `DELETE /chat/{session_id}` - Clear a chat session
- `GET /chat/sessions` - List active chat sessions

## 📁 Project Structure

```
rag-ingestion-retrival-template/
├── rag_ingestion.py              # Main RAG ingestion script
├── rag_service.py                # FastAPI service for database operations
├── rag_retrival.py               # Conversational RAG API service
├── requirements.txt              # Python dependencies
├── schema_rag_ingestion.sql      # Database schema for RAG system
├── schema_rag_ingestion_updated.sql  # Updated database schema
├── .env                          # Environment variables (create from template)
├── .gitignore                    # Git ignore rules
├── README.md                     # This file
└── venv/                         # Virtual environment (not tracked)
```

## 🔧 Key Functions

### Core RAG Functions
- `get_s3_client()`: Creates AWS S3 client
- `list_s3_files()`: Lists files in S3 bucket
- `extract_text_from_pdf()`: Extracts text from PDF content
- `chunk_text()`: Splits text into overlapping chunks
- `embed_chunks()`: Generates embeddings using OpenAI

### Database Functions
- `get_db_pool()`: Creates PostgreSQL connection pool with pgvector support
- `get_or_create_embedding_model()`: Manages embedding model configurations
- `create_ingestion_run()`: Tracks ingestion process runs
- `get_or_create_source()`: Manages source information
- `create_document()`: Stores document metadata and content
- `create_chunks()`: Stores text chunks with metadata
- `create_chunk_embeddings()`: Stores vector embeddings
- `query_chunks()`: Performs vector similarity search

### Conversational RAG Functions
- `fetch_relevant_documents()`: Retrieves relevant documents for queries
- `get_or_create_session()`: Manages chat sessions with memory
- `ServiceRetriever`: Custom retriever for the conversational chain

### Main Functions
- `process_s3_file_to_api()`: Processes a single S3 file end-to-end
- `main()`: Orchestrates the entire RAG pipeline with database integration

## ⚙️ Configuration Options

- **Chunk Size**: Default 300 characters (configurable in `chunk_text()`)
- **Chunk Overlap**: Default 60 characters (configurable in `chunk_text()`)
- **Embedding Model**: OpenAI text-embedding-ada-002 (1536 dimensions)
- **AWS Region**: Configurable via environment variable
- **Database Connection**: PostgreSQL with pgvector extension
- **Similarity Threshold**: Configurable in search functions (default 0.7 for all search operations)
- **Search Limit**: Configurable number of results (default 10)
- **Chat Memory**: Maintains conversation history with window of 5 messages

## 🛡️ Error Handling

The system includes comprehensive error handling for:
- AWS authentication issues
- S3 access problems
- PDF processing errors
- Database connection issues
- Network connectivity issues
- Invalid file formats
- Embedding generation failures
- Database transaction rollbacks
- API service failures

## 🔒 Security Notes

- Never commit your `.env` file with real credentials
- Use IAM roles when possible instead of access keys
- Regularly rotate your AWS credentials
- Follow AWS security best practices

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🆘 Support

For issues and questions, please open an issue on the GitHub repository.

## 🔑 OpenAI Setup

Ensure you have:
1. OpenAI API key with access to text-embedding-ada-002 model
2. Sufficient API credits for embedding generation
3. Proper IAM permissions for S3 access (for document ingestion)

## 📊 Example Usage

### Chat with the RAG System

```bash
curl -X POST "http://localhost:8001/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What is the main topic of the documents?",
       "session_id": "user123"
     }'
```

### Query for Similar Chunks

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "machine learning algorithms",
       "similarity_threshold": 0.7,
       "max_results": 5
     }'
```

## 🎯 Use Cases

This template is perfect for:
- Document search and retrieval systems
- Question-answering applications
- Knowledge base construction
- Content recommendation systems
- Research assistance tools
- Customer support chatbots
- Legal document analysis
- Academic paper processing

---

**Template Repository**: This is a template repository for RAG ingestion and retrieval systems. Use this as a starting point for your own RAG applications.
=======
# rag-ingestion-retrival-template
Use this anywhere in the application for ingestion and retrival
>>>>>>> 14c84719192ab5e15b58f85f8e8172a7976c7857
