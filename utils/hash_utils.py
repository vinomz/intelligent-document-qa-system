import hashlib
import os

class HashUtils:
    @staticmethod
    def md5_file(path: str) -> str:
        """Compute MD5 hash of a file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        hasher = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def sha256_file(path: str) -> str:
        """Compute SHA-256 hash of a file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def md5_text(text: str) -> str:
        """Compute MD5 hash of a text string."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    @staticmethod
    def sha256_text(text: str) -> str:
        """Compute SHA-256 hash of a text string."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
