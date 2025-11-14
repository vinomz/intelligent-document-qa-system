from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import settings

class EmbeddingsFactory:
    @staticmethod
    def create():
        return GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL)
