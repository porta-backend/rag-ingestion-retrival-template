# RAG (Retrieval-Augmented Generation) System with AWS Bedrock

A Python-based RAG system that processes PDF documents from AWS S3, extracts text, creates embeddings using Amazon Bedrock's Titan embedding model, and displays the vector representations.

## Features

- **S3 Integration**: Automatically downloads and processes PDF files from AWS S3 buckets
- **PDF Text Extraction**: Extracts text content from PDF documents using PyPDF2
- **Text Chunking**: Splits large documents into manageable chunks with configurable overlap
- **AWS Bedrock Embeddings**: Uses Amazon Titan embedding model to generate vector representations
- **Vector Display**: Shows detailed embedding vectors with statistics and analysis
- **Error Handling**: Comprehensive error handling for robust operation

## Prerequisites

- Python 3.7+
- AWS Account with Bedrock access
- AWS S3 bucket with PDF files
- AWS credentials configured

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

2. Edit `.env` file with your AWS credentials and configuration:
```env
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=us-east-1
AWS_S3_BUCKET=your-bucket-name
```

## Usage

Run the RAG system:
```bash
python rag.py
```

The script will:
1. Connect to your S3 bucket
2. List all PDF files
3. Download and extract text from each PDF
4. Split text into chunks
5. Generate embeddings using Amazon Bedrock Titan model
6. Display embedding vectors and statistics

## Output

The system provides:
- **Console Output**: Detailed information about processed files, chunks, and embedding statistics
- **Vector Display**: Shows the first few embedding vectors with their numerical values
- **Statistics**: Min, max, mean, and magnitude of embedding vectors

## Project Structure

```
porta-rag/
├── rag.py              # Main RAG implementation
├── requirements.txt    # Python dependencies
├── .env.example       # Environment variables template
├── .gitignore         # Git ignore rules
├── README.md          # This file
└── venv/              # Virtual environment (not tracked)
```

## Key Functions

- `get_s3_client()`: Creates AWS S3 client
- `list_s3_files()`: Lists files in S3 bucket
- `extract_text_from_pdf()`: Extracts text from PDF content
- `chunk_text()`: Splits text into overlapping chunks
- `embed_chunks()`: Generates embeddings using Bedrock
- `main()`: Orchestrates the entire RAG pipeline

## Configuration Options

- **Chunk Size**: Default 1000 characters (configurable in `chunk_text()`)
- **Chunk Overlap**: Default 200 characters (configurable in `chunk_text()`)
- **Embedding Model**: Amazon Titan Embed Text v1
- **AWS Region**: Configurable via environment variable

## Error Handling

The system includes comprehensive error handling for:
- AWS authentication issues
- S3 access problems
- PDF processing errors
- Network connectivity issues
- Invalid file formats

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
