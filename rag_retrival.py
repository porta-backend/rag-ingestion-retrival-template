import os
import httpx
from typing import Dict, Optional, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from langchain_aws.chat_models import ChatBedrockConverse
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

class QueryInput(BaseModel):
    query: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    source_documents: List[Dict]
    chat_history: List[Dict]

class RetrievalConfig(BaseModel):
    max_results: int = 10
    similarity_threshold: float = 0.5
    model_name: str = "amazon.titan-embed-text-v1"

def fetch_relevant_documents(query: str, config: RetrievalConfig) -> List[Document]:
    try:
        print(f"[RETRIEVAL] Query: {query[:100]}...")
        print(f"[RETRIEVAL] Config: threshold={config.similarity_threshold}, max_results={config.max_results}, model={config.model_name}")
        
        with httpx.Client(base_url=API_BASE_URL, timeout=30.0) as client:
            response = client.post("/query", json={
                "query": query,
                "similarity_threshold": config.similarity_threshold,
                "max_results": config.max_results,
                "model_name": config.model_name
            })
            response.raise_for_status()
            result = response.json()
        
        print(f"[RETRIEVAL] API returned {len(result.get('results', []))} results")
        
        documents = []
        for i, chunk in enumerate(result.get("results", [])):
            print(f"[RETRIEVAL] Result {i+1}: similarity={chunk.get('similarity_score', 'N/A'):.4f}, content={chunk.get('content', '')[:50]}...")
            
            metadata = {
                "chunk_id": chunk.get("chunk_id"),
                "similarity_score": chunk.get("similarity_score"),
                "document_title": chunk.get("document_title"),
                "document_author": chunk.get("document_author"),
                "source_uri": chunk.get("source_uri"),
                "chunk_index": chunk.get("chunk_index"),
                "token_count": chunk.get("token_count")
            }
            if chunk.get("metadata"):
                metadata.update(chunk["metadata"])
            
            doc = Document(page_content=chunk.get("content", ""), metadata=metadata)
            documents.append(doc)
        
        return documents
    except Exception as e:
        print(f"[RETRIEVAL] Error in service retrieval: {str(e)}")
        return []

session_state: Dict[str, Dict] = {}

def get_or_create_session(session_id: str):
    if session_id not in session_state:
        llm = ChatBedrockConverse(
            model_id="amazon.nova-lite-v1:0",
            region_name=AWS_REGION,
        )
        
        custom_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based ONLY on the provided context documents. 

IMPORTANT RULES:
1. ONLY use information from the provided context documents below
2. If the answer is not in the context documents, say "I don't have enough information in the provided documents to answer this question."
3. Answer everything in less than 100 words and bullet points are only allowed
4. Everything should be in the markdown format

Context Documents:
{context}

Chat History:
{chat_history}

Question: {question}

Answer based ONLY on the context documents above:
""")
        
        retriever_config = RetrievalConfig()
        retriever_instance = ServiceRetriever(API_BASE_URL, retriever_config)
        memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, output_key="answer", k=5)
        
        conversational_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever_instance,
            memory=memory,
            return_source_documents=True,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": custom_prompt}
        )
        
        session_state[session_id] = {
            "chain": conversational_chain,
            "memory": memory
        }
    return session_state[session_id]["chain"], session_state[session_id]["memory"]

class ServiceRetriever(BaseRetriever):
    api_base_url: str
    config: RetrievalConfig
    
    def __init__(self, api_base_url: str, config: RetrievalConfig, **kwargs):
        super().__init__(api_base_url=api_base_url, config=config, **kwargs)
        self.api_base_url = api_base_url
        self.config = config

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return fetch_relevant_documents(query, self.config)

app = FastAPI(
    title="Conversational RAG Chatbot API",
    description="An API that takes a user query and returns a conversational response from a RAG system.",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "conversational-rag-api"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(input_data: QueryInput):
    if not input_data.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty.")
        
    session_id = input_data.session_id or "default_session"
    
    try:
        print(f"[CHAT] Processing query: {input_data.query}")
        chain, memory = get_or_create_session(session_id)
        
        config = RetrievalConfig()
        retriever_instance = ServiceRetriever(API_BASE_URL, config)
        test_docs = retriever_instance._get_relevant_documents(input_data.query)
        print(f"[CHAT] Direct retrieval test: {len(test_docs)} documents")
        
        response = chain.invoke({"question": input_data.query})
        
        print(f"[CHAT] Chain response keys: {response.keys()}")
        print(f"[CHAT] Source documents count: {len(response.get('source_documents', []))}")
        
        source_documents = [doc.metadata for doc in response.get("source_documents", [])]
        chat_history_list = [
            {"type": msg.type, "content": str(msg.content)} 
            for msg in memory.chat_memory.messages
        ]
        
        return ChatResponse(
            response=response["answer"],
            source_documents=source_documents,
            chat_history=chat_history_list
        )
    except Exception as e:
        print(f"[CHAT] Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred: {str(e)}"
        )

@app.delete("/chat/{session_id}")
async def clear_session(session_id: str):
    if session_id in session_state:
        del session_state[session_id]
        return {"message": f"Session {session_id} cleared successfully"}
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

@app.get("/chat/sessions")
async def list_sessions():
    return {"active_sessions": list(session_state.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)