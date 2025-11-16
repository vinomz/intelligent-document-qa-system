from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import settings
from utils.performance_calc import Metrics
import time

metrics = Metrics()

class TimedGoogleEmbeddings(GoogleGenerativeAIEmbeddings):
    """Measure embedding latency per query."""

    def embed_query(self, text):
        t0 = time.time()
        vec = super().embed_query(text)
        ms = (time.time() - t0) * 1000

        metrics.embedding.record(ms)   # store embedding p50/p95/p99
        return vec

    def embed_documents(self, texts):
        t0 = time.time()
        vecs = super().embed_documents(texts)
        ms = (time.time() - t0) * 1000

        metrics.embedding.record(ms)
        return vecs

class EmbeddingsFactory:
    @staticmethod
    def create():
        return TimedGoogleEmbeddings(model=settings.EMBEDDING_MODEL)
