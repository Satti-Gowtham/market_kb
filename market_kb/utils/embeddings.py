import logging
from typing import List
from langchain_ollama import OllamaEmbeddings as OllamaEmbeddingsLangchain

logger = logging.getLogger(__name__)

class OllamaEmbeddings:
    def __init__(self, model: str = "nomic-embed-text", url: str = "http://localhost:11434"):
        self.model = model
        self.url = url
        self.client = OllamaEmbeddingsLangchain(model=self.model, base_url=self.url)

    async def embed_text(self, text: str) -> List[float]:
        """Create embedding for a single text"""
        try:
            logger.info(f"Embedding text: {text}...")
            embedding = self.client.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            raise

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts"""
        try:
            logger.info(f"Embedding {len(texts)} texts")
            embeddings = self.client.embed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise

