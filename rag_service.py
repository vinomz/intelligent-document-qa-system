# rag_service.py

import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from typing import Any, Dict
from config import settings

os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
DATA_PATH = settings.DEFAULT_DOCS_PATH
CHROMA_PATH = settings.DEFAULT_CHROMA_PATH

# Global RAG Chain variable
RAG_CHAIN = None

def get_rag_chain(retriever: Any) -> Any:
    """Initializes and returns the RAG chain."""
    
    # 1. Initialize LLM (Gemini)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

    # 2. Define the Prompt Template
    system_prompt = (
        "You are an expert AI Knowledge Assistant for an internal team. "
        "Your task is to answer the user's question ONLY based on the "
        "provided context documents. Do not use external knowledge. If the answer "
        "is not found in the context, state clearly: 'I am sorry, the answer "
        "is not available in the internal knowledge base.'"
        "\n\nContext: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # 3. Create the LangChain pipeline
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain

def load_or_create_vector_store():
    """Loads ChromaDB or creates it if the path doesn't exist."""
    
    # 1. Create Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # Check if the vector store already exists
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        print("Loading existing ChromaDB from persistence...")
        # Load the persisted vector store
        vector_store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
    else:
        print("ChromaDB not found or empty. Starting indexing process...")
        # Perform indexing (Data Ingestion)
        
        # 2. Load Data (ensure internal_docs directory exists with files)
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
            print(f"Created directory: {DATA_PATH}. Please add documents.")
            return None

        loader = PyPDFDirectoryLoader(DATA_PATH)
        documents = loader.load()

        # 3. Split Documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Indexing {len(chunks)} chunks...")

        # 4. Store in ChromaDB
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH
        )
        vector_store.persist()
        print(f"Indexing complete. Knowledge base saved to {CHROMA_PATH}")

    return vector_store

def initialize_assistant():
    """Initializes the RAG components and sets the global RAG_CHAIN."""
    global RAG_CHAIN
    
    vector_store = load_or_create_vector_store()
    
    if vector_store:
        # Create a retriever from the vector store
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        RAG_CHAIN = get_rag_chain(retriever)
        print("RAG Chain initialized and ready.")
        return True
    return False

async def query_assistant(query: str) -> Dict[str, Any]:
    """Runs the user query through the RAG chain."""
    global RAG_CHAIN
    
    if not RAG_CHAIN:
        raise Exception("RAG Chain is not initialized. Check logs for errors.")

    response = RAG_CHAIN.invoke({"input": query})
    
    # Extract the final answer and source documents
    answer = response["answer"]
    sources = [
        {"source": doc.metadata.get('source', 'N/A'), "page": str(doc.metadata.get('page', 'N/A'))}
        for doc in response.get("context", [])
    ]
    
    return {"answer": answer, "sources": sources}

# Call this function once when the API starts
initialize_assistant()