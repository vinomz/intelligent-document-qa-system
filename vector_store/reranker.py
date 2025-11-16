from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

class Reranker:
    def __init__(self, retriever, top_n=3):
        self.retriever = retriever
        self.reranker_model_name = "BAAI/bge-reranker-base"

        # Load HF CrossEncoder
        self.reranker_model = HuggingFaceCrossEncoder(
            model_name=self.reranker_model_name
        )

        # Build compressor
        self.compressor = CrossEncoderReranker(
            model=self.reranker_model,
            top_n=top_n
        )

        # Build contextual compressor retriever
        self.compression_retriever = ContextualCompressionRetriever(
            base_retriever=self.retriever,
            base_compressor=self.compressor
        )
    
    def get_retriever(self):
        return self.compression_retriever
        