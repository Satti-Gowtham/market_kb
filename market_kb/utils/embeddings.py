import logging
import aiohttp
from typing import List

logger = logging.getLogger(__name__)

class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text", url: str = "http://localhost:11434"):
        self.model = model
        self.url = url
        self.api_url = f"{url}/api/embeddings"

    async def embed_text(self, text: str) -> List[float]:
        """Create embedding for a single text"""
        try:
            logger.info(f"Embedding text: {text[:100]}...")
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "prompt": text
                }
                async with session.post(self.api_url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error: {error_text}")
                    result = await response.json()
                    return result["embedding"]
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            raise

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts"""
        try:
            logger.info(f"Embedding {len(texts)} texts")
            embeddings = []
            async with aiohttp.ClientSession() as session:
                for text in texts:
                    payload = {
                        "model": self.model,
                        "prompt": text
                    }
                    async with session.post(self.api_url, json=payload) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise Exception(f"Ollama API error: {error_text}")
                        result = await response.json()
                        embeddings.append(result["embedding"])
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise

