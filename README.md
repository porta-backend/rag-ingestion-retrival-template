# RAG (Retrieval-Augmented Generation) System with Database Integration

A comprehensive Python-based RAG system that processes PDF documents from AWS S3, extracts text, creates embeddings using Amazon Bedrock's Titan embedding model, and stores everything in a PostgreSQL database with pgvector support for efficient similarity search.

## Features

- **S3 Integration**: Automatically downloads and processes PDF files from AWS S3 buckets
- **PDF Text Extraction**: Extracts text content from PDF documents using PyPDF2
- **Text Chunking**: Splits large documents into manageable chunks with configurable overlap
- **AWS Bedrock Embeddings**: Uses Amazon Titan embedding model to generate vector representations
- **PostgreSQL + pgvector Storage**: Stores documents, chunks, and embeddings in a structured database
- **Vector Similarity Search**: Efficient similarity search using pgvector's HNSW indexing
- **Ingestion Tracking**: Tracks ingestion runs with status and metadata
- **Comprehensive Metadata**: Stores rich metadata for sources, documents, and chunks
- **Error Handling**: Comprehensive error handling for robust operation

## Prerequisites

- Python 3.7+
- AWS Account with Bedrock access
- AWS S3 bucket with PDF files
- AWS credentials configured
- PostgreSQL database with pgvector extension
- Database credentials configured

## Installation

1. Clone the repository:
```bash
git clone https://github.com/pavankota20/porta-rag.git
cd porta-rag
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

## Configuration

1. Copy the environment example file:
```bash
cp .env.example .env
```

2. Edit `.env` file with your AWS credentials and database configuration:
```env
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=us-east-1
AWS_S3_BUCKET=your-bucket-name

# Database Configuration (for API service)
HOST=your-database-host
PORT=5432
USERNAME=your-database-username
PASSWORD=your-database-password
DATABASE=your-database-name

# API Configuration (for RAG client)
API_BASE_URL=http://localhost:8000
```

## Database Setup

Before running the system, ensure your PostgreSQL database has the required schema. The system expects the following tables:

- `embedding_models`: Stores embedding model configurations
- `ingestion_runs`: Tracks ingestion process runs
- `sources`: Stores source information (S3 buckets, files, etc.)
- `documents`: Stores document metadata and content
- `chunks`: Stores text chunks with full-text search support
- `chunk_embeddings`: Stores vector embeddings with pgvector support

See the database schema in your database setup for the complete table definitions.

## Usage

### 1. Start the API Service

First, start the database API service:

```bash
python start_api.py
```

This will start the FastAPI service on `http://localhost:8000`. You can view the API documentation at `http://localhost:8000/docs`.

### 2. Run the Main Ingestion Process

In a new terminal, run the RAG ingestion:

```bash
python rag.py
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
   - Generate embeddings using Amazon Bedrock Titan model
   - Store everything in the database via API
6. Update ingestion run status via API

### 3. API Endpoints

The API service provides the following endpoints:
- `GET /health` - Health check
- `POST /embedding-models` - Create/get embedding models
- `POST /ingestion-runs` - Create ingestion runs
- `PUT /ingestion-runs/{run_id}` - Update ingestion run status
- `POST /sources` - Create/get sources
- `POST /documents` - Create documents
- `POST /chunks` - Create chunks
- `POST /embeddings` - Create embeddings

## Output

The system provides:
- **Console Output**: Detailed information about processed files, chunks, and embedding statistics
- **Database Storage**: All documents, chunks, and embeddings stored in PostgreSQL
- **Ingestion Tracking**: Complete audit trail of ingestion runs and their status
- **Vector Search Results**: Similarity scores and document metadata for search queries

## Project Structure

```
rag_highlevel/
├── rag.py              # Main RAG implementation (uses API for database)
├── api_service.py      # FastAPI service for database operations
├── start_api.py        # Script to start the API service
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (create from template)
├── .gitignore         # Git ignore rules
├── README.md          # This file
└── venv/              # Virtual environment (not tracked)
```

## Key Functions

### Core RAG Functions
- `get_s3_client()`: Creates AWS S3 client
- `list_s3_files()`: Lists files in S3 bucket
- `extract_text_from_pdf()`: Extracts text from PDF content
- `chunk_text()`: Splits text into overlapping chunks
- `embed_chunks()`: Generates embeddings using Bedrock

### Database Functions
- `get_db_connection()`: Creates PostgreSQL connection with pgvector support
- `get_or_create_embedding_model()`: Manages embedding model configurations
- `create_ingestion_run()`: Tracks ingestion process runs
- `get_or_create_source()`: Manages source information
- `create_document()`: Stores document metadata and content
- `create_chunks()`: Stores text chunks with metadata
- `create_chunk_embeddings()`: Stores vector embeddings
- `search_similar_chunks()`: Performs vector similarity search
- `get_ingestion_run_summary()`: Retrieves ingestion run statistics

### Main Functions
- `process_s3_file_to_database()`: Processes a single S3 file end-to-end
- `main()`: Orchestrates the entire RAG pipeline with database integration

## Configuration Options

- **Chunk Size**: Default 1000 characters (configurable in `chunk_text()`)
- **Chunk Overlap**: Default 200 characters (configurable in `chunk_text()`)
- **Embedding Model**: Amazon Titan Embed Text v1 (1536 dimensions)
- **AWS Region**: Configurable via environment variable
- **Database Connection**: PostgreSQL with pgvector extension
- **Similarity Threshold**: Configurable in search functions (default 0.7)
- **Search Limit**: Configurable number of results (default 5)

## Error Handling

The system includes comprehensive error handling for:
- AWS authentication issues
- S3 access problems
- PDF processing errors
- Database connection issues
- Network connectivity issues
- Invalid file formats
- Embedding generation failures
- Database transaction rollbacks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Support

For issues and questions, please open an issue on the GitHub repository.

## AWS Bedrock Setup

Ensure your AWS account has:
1. Bedrock service enabled
2. Access to Amazon Titan embedding models
3. Proper IAM permissions for S3 and Bedrock access

## Security Notes

- Never commit your `.env` file with real credentials
- Use IAM roles when possible instead of access keys
- Regularly rotate your AWS credentials
- Follow AWS security best practices
