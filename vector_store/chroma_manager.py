import os
from langchain_chroma import Chroma
from .embeddings import EmbeddingsFactory
from .loader import DocumentLoader
from utils.logger import get_logger
from utils.hash_utils import HashUtils
from preprocess.text_cleaner import clean_text

class ChromaManager:
    def __init__(self, data_path, chroma_path):
        self.data_path = data_path
        self.chroma_path = chroma_path
        self.db = None
        self.logger = get_logger("chroma_manager")
        # self.conflict_mode = "replace"

    def load_db(self):
        try:
            embeddings = EmbeddingsFactory.create()

            self.logger.info("Loading Chroma DB...")
            self.db = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=embeddings
            )

            self.db.similarity_search("warm-up", k=1)

            self.logger.info("Chroma DB is ready.")
            
            return self.db
        except Exception as err:
            self.logger.error(f"Failed to load Chroma DB | Error ==> {err}")
            return None
    
    def _delete_by_filename(self, filename: str):
        self.db.delete(where={"filename": filename})
        self.logger.info(f"Deleted all chunks for file: {filename}")
    
    def generate_document_id(self, path: str) -> str:
        """Stable document_id based on file path."""
        return HashUtils.md5_text(path)

    def update_index(self):
        try:
            self.logger.info("Updating Chroma DB...")
            existing = self.db.get(include=["metadatas"])
            existing_metas = existing["metadatas"]
            self.logger.debug(f"Existing Metas: {existing_metas}")

            existing_hashes = {m.get("file_hash") for m in existing_metas if m.get("file_hash")}
            self.logger.debug(f"Existing Hashes: {existing_hashes}")
            existing_filenames = {m.get("filename") for m in existing_metas if m.get("filename")}
            self.logger.debug(f"Existing File Name: {existing_filenames}")

            skip_filenames = []
            filehashes = {}

            for filename in os.listdir(self.data_path):
                filepath = os.path.join(self.data_path, filename)

                if not filename.lower().endswith((".pdf", ".txt", ".docx")):
                    continue

                filename = os.path.basename(filepath)
                filehash = HashUtils.md5_file(filepath)
                filehashes[filename] = filehash

                if filehash in existing_hashes:
                    self.logger.info(f"Skipping unchanged file: {filename}")
                    skip_filenames.append(filename)
                    continue
                
                # # Conflict: same filename, different content
                # if filename in existing_filenames:
                #     if self.conflict_mode == "ignore":
                #         self.logger.info(f"Ignored new version of file: {filename}")
                #         skip_filenames.append(filename)
                #         continue

                #     if self.conflict_mode == "replace":
                #         self.logger.info(f"Replacing existing file: {filename}")
                #         self._delete_by_filename(filename)
            
            loader = DocumentLoader(self.data_path)
            documents = loader.load_documents()

            new_docs = []
            for doc in documents:
                filepath = doc.metadata["source"]
                filename = os.path.basename(filepath)
                if filename in skip_filenames:
                    continue
                
                # Clean + normalize text here
                doc.page_content = clean_text(doc.page_content)
                # Add metadata
                doc.metadata["filename"] = filename
                doc.metadata["file_hash"] = filehashes.get(filename)
                doc.metadata["document_id"] = self.generate_document_id(filepath)

                if "page" not in doc.metadata:
                    doc.metadata["page"] = 0

                new_docs.append(doc)
            
            if not new_docs:
                self.logger.info("No new docs to index.")
                return "No new docs to index."
            
            chunks = loader.split_documents(new_docs)

            for idx, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = idx
                
            self.db.add_documents(chunks)
            self.logger.info(f"Indexed {len(chunks)} chunks.")

            return "Sucessfully re-indexed the DB"
        except Exception as err:
            self.logger.error(f"Failed to re-index | Error ==> {err}")
            return f"Failed to re-index | Error ==> {err}"
