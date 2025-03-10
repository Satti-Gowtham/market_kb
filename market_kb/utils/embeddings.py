import logging
import aiohttp
from typing import List
import json

logger = logging.getLogger(__name__)

class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text", url: str = "http://localhost:11434"):
        self.model = model
        self.url = url
        self.api_url = f"{url}/api/embeddings"
        logger.info(f"Initialized OllamaEmbedder with model: {model}, API URL: {self.api_url}")

    async def _check_embedding_quality(self, embedding):
        """Check if the embedding looks valid"""
        if not embedding:
            logger.warning("Empty embedding received")
            return False
            
        if not isinstance(embedding, list):
            logger.warning(f"Embedding is not a list: {type(embedding)}")
            return False
            
        if len(embedding) < 10:
            logger.warning(f"Embedding is suspiciously short: {len(embedding)} dimensions")
            return False
            
        nonzero_values = sum(1 for x in embedding if abs(x) > 1e-6)
        if nonzero_values == 0:
            logger.warning("Embedding contains all zeros or near-zeros")
            return False
            
        return True

    async def embed_text(self, text: str) -> List[float]:
        """Create embedding for a single text"""
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * 768  # Default embedding dimension 
            
        try:
            logger.info(f"Embedding text: {text[:100]}...")
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "prompt": text
                }
                
                try:
                    async with session.post(self.api_url, json=payload) as response:
                        status = response.status
                        
                        if status != 200:
                            error_text = await response.text()
                            logger.error(f"Ollama API error ({status}): {error_text}")
                            return [0.0] * 768
                        
                        result = await response.json()
                        
                        if "embedding" not in result:
                            logger.error(f"No embedding in response. Keys: {list(result.keys())}")
                            return [0.0] * 768
                            
                        embedding = result["embedding"]
                        
                        quality_ok = await self._check_embedding_quality(embedding)
                        if not quality_ok:
                            logger.warning("Low quality embedding received, using fallback")
                            return [0.0] * 768
                            
                        logger.debug(f"Embedding length: {len(embedding)}")
                        return embedding
                except aiohttp.ClientError as e:
                    logger.error(f"HTTP error connecting to Ollama: {str(e)}")
                    return [0.0] * 768
                    
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            return [0.0] * 768

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts"""
        if not texts:
            logger.warning("Empty texts list provided for embedding")
            return []
            
        try:
            logger.info(f"Embedding {len(texts)} texts")
            embeddings = []
            
            success_count = 0
            
            async with aiohttp.ClientSession() as session:
                for i, text in enumerate(texts):
                    if not text or not text.strip():
                        logger.warning(f"Empty text at position {i}")
                        embeddings.append([0.0] * 768)
                        continue
                        
                    payload = {
                        "model": self.model,
                        "prompt": text
                    }
                    
                    try:
                        async with session.post(self.api_url, json=payload) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                logger.error(f"Ollama API error for text {i}: {error_text}")
                                embeddings.append([0.0] * 768)
                                continue
                                
                            result = await response.json()
                            
                            if "embedding" not in result:
                                logger.error(f"No embedding in response for text {i}")
                                embeddings.append([0.0] * 768)
                                continue
                                
                            embedding = result["embedding"]
                            
                            quality_ok = await self._check_embedding_quality(embedding)
                            if not quality_ok:
                                logger.warning(f"Low quality embedding received for text {i}")
                                embeddings.append([0.0] * 768)
                                continue
                                
                            embeddings.append(embedding)
                            success_count += 1
                    except Exception as e:
                        logger.error(f"Error embedding text {i}: {str(e)}")
                        embeddings.append([0.0] * 768)
            
            logger.info(f"Embedding success rate: {success_count}/{len(texts)}")
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            return [[0.0] * 768 for _ in range(len(texts))]

