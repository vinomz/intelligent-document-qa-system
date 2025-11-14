from vector_store.chroma_manager import ChromaManager
from rag.chain_builder import RAGChainBuilder
from rag.prompts import fallback_answer

class AssistantService:
    def __init__(self, data_path, chroma_path):
        self.data_path = data_path
        self.chroma_path = chroma_path
        self.chain = None

    def initialize(self):
        manager = ChromaManager(self.data_path, self.chroma_path)
        vector_store = manager.load_or_create()
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        self.chain = RAGChainBuilder(retriever).build()

    async def query(self, query: str):
        if not self.chain:
            raise Exception("RAG not initialized")

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

        return {"answer": answer, "sources": sources}
