from rerankers import Reranker
from langchain_core.runnables import RunnableLambda
from utils.logger import get_logger

from utils.performance_calc import Metrics
import time

from config import settings

metrics = Metrics()

logger = get_logger("reranker")

class RerankerWrapper:
    def __init__(self, retriever, top_n=settings.RERANKER_TOP_N):
        self.retriever = retriever
        self.top_n = top_n
        self.model = Reranker(settings.RERANKER_MODEL, model_type=settings.RERANKER_TYPE)

    def rerank(self, query):
        try:
            docs = self.retriever.invoke(query)
            if len(docs) == 0:
                return []
                
            texts = [d.page_content for d in docs]

            t0 = time.time()
            ranked_results = self.model.rank(query, texts)

            logger.info(f"Ranked results: {ranked_results}")

            reranked_docs = []
            for res in ranked_results.results[:self.top_n]:
                original_doc = docs[res.document.doc_id]
                reranked_docs.append(original_doc)
            metrics.rerank.record((time.time() - t0) * 1000)

            return reranked_docs
        except Exception as err:
            logger.error(f"Failed to rerank | Error ==> {err}")

    def get_retriever(self):
        # Return a Runnable retriever
        return RunnableLambda(lambda q: self.rerank(q))

# class Reranker:
#     def __init__(self, retriever, top_n=3):
#         from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
#         from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
#         from langchain_community.cross_encoders import HuggingFaceCrossEncoder

#         self.retriever = retriever
#         self.reranker_model_name = "BAAI/bge-reranker-base"

#         # Load HF CrossEncoder
#         self.reranker_model = HuggingFaceCrossEncoder(
#             model_name=self.reranker_model_name
#         )

#         # Build compressor
#         self.compressor = CrossEncoderReranker(
#             model=self.reranker_model,
#             top_n=top_n
#         )

#         # Build contextual compressor retriever
#         self.compression_retriever = ContextualCompressionRetriever(
#             base_retriever=self.retriever,
#             base_compressor=self.compressor
#         )
    
#     def get_retriever(self):
#         return self.compression_retriever
        