import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def smart_loader(path):
    lower = path.lower()
    if lower.endswith(".pdf"):
        return PyPDFLoader(path)
    if lower.endswith(".docx"):
        return Docx2txtLoader(path)
    if lower.endswith(".txt"):
        return TextLoader(path, encoding="utf-8")
    return None

class DocumentLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_documents(self):
        loader = DirectoryLoader(
            self.data_path,
            loader_cls=smart_loader,
            glob="**/*",
            show_progress=True
        )
        return loader.load()

    def split_documents(self, docs):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return splitter.split_documents(docs)
