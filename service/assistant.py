from vector_store.chroma_manager import ChromaManager
from vector_store.retriever import FastChromaRetriever
from rag.chain_builder import RAGChainBuilder
from rag.prompts import fallback_answer
from utils.logger import get_logger
from config import settings

class AssistantService:
    def __init__(self, data_path, chroma_path):
        self.data_path = data_path
        self.chroma_path = chroma_path
        self.use_reranker = True
        self.chain = None
        self.manager = None

        self.logger = get_logger("assistant_service")

    def initialize(self):
        self.manager = ChromaManager(self.data_path, self.chroma_path)
        vector_store = self.manager.load_db()
        retriever = FastChromaRetriever(vector_store, k=settings.VECTOR_STORE_K)

        if self.use_reranker:
            from vector_store.reranker import RerankerWrapper
            retriever = RerankerWrapper(retriever, top_n=settings.RERANKER_TOP_N).get_retriever()

        self.chain = RAGChainBuilder(retriever).build()
    
    async def reindex(self):
        result = self.manager.update_index()
        return result

    async def query(self, query: str):
        if not self.chain:
            raise Exception("RAG not initialized")

        try:
            result = self.chain.invoke(query)
            answer = result["response"].content

            if answer == fallback_answer:
                return {"answer": answer, "sources": [], "total_tokens": 0}

            sources = [
                {
                    "source": doc.metadata.get("source", "N/A"),
                    "page": str(doc.metadata.get("page", "N/A"))
                }
                for doc in result["retrieved_docs"]
            ]
            
            usage_metadata = result["response"].usage_metadata
            total_tokens = usage_metadata.get("total_tokens", 0)

            self.logger.info(f"Query: {query} | Answer: {answer} | Sources: {sources} | Usage: {total_tokens}")

            return {"answer": answer, "sources": sources, "total_tokens": total_tokens}

        except Exception as e:
            self.logger.error(f"Error in query: {e}")
            return {"answer": fallback_answer, "sources": [], "total_tokens": 0}
