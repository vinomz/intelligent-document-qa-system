class FastChromaRetriever:
    def __init__(self, chroma_db, k=10):
        self.chroma = chroma_db
        self.k = k

    def invoke(self, query):
        # MUCH faster than as_retriever()
        results = self.chroma.similarity_search_with_relevance_scores(
            query,
            k=self.k
        )
        # results -> [(Document, score), ...]
        # For RAG you only need the Document
        return [doc for doc, score in results]
