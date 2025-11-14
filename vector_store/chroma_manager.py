import os
from langchain_chroma import Chroma
from .embeddings import EmbeddingsFactory
from .loader import DocumentLoader

class ChromaManager:
    def __init__(self, data_path, chroma_path):
        self.data_path = data_path
        self.chroma_path = chroma_path

    def load_or_create(self):
        embeddings = EmbeddingsFactory.create()

        # If DB already exists → load
        if os.path.exists(self.chroma_path) and os.listdir(self.chroma_path):
            return Chroma(
                persist_directory=self.chroma_path,
                embedding_function=embeddings
            )

        # Else create
        loader = DocumentLoader(self.data_path)
        documents = loader.load_documents()
        chunks = loader.split_documents(documents)

        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.chroma_path
        )
        return db
