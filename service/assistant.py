from vector_store.chroma_manager import ChromaManager
from vector_store.retriever import FastChromaRetriever
from rag.chain_builder import RAGChainBuilder
from rag.prompts import fallback_answer
from utils.logger import get_logger

class AssistantService:
    def __init__(self, data_path, chroma_path):
        self.data_path = data_path
        self.chroma_path = chroma_path
        self.use_reranker = False
        self.chain = None
        self.manager = None

        self.logger = get_logger("assistant_service")

    def initialize(self):
        self.manager = ChromaManager(self.data_path, self.chroma_path)
        vector_store = self.manager.load_db()
        retriever = FastChromaRetriever(vector_store, k=10)

        if self.use_reranker:
            from vector_store.reranker import Reranker
            retriever = Reranker(retriever).get_retriever()

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
                return {"answer": answer, "sources": []}

            sources = [
                {
                    "source": doc.metadata.get("source", "N/A"),
                    "page": str(doc.metadata.get("page", "N/A"))
                }
                for doc in result["retrieved_docs"]
            ]
            
            self.logger.info(f"Query: {query} | Answer: {answer} | Sources: {sources}")

            return {"answer": answer, "sources": sources}

        except Exception as e:
            self.logger.error(f"Error in query: {e}")
            return {"answer": fallback_answer, "sources": []}
