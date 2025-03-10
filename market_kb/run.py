import pytz
from typing import Dict, Any
from datetime import datetime
import json
import uuid
import math
from naptha_sdk.schemas import KBRunInput
from naptha_sdk.storage.storage_client import StorageClient
from naptha_sdk.storage.schemas import (
    CreateStorageRequest, 
    ReadStorageRequest, 
    DeleteStorageRequest,
    ListStorageRequest,
    DatabaseReadOptions
)
from naptha_sdk.user import sign_consumer_id
from naptha_sdk.utils import get_logger
from dotenv import load_dotenv

from market_kb.schemas import InputSchema, RetrievedMemory
from market_kb.utils.chunker import SemanticChunker
from market_kb.utils.embeddings import OllamaEmbedder

load_dotenv()

logger = get_logger(__name__)

class MarketKB:
    """Market Knowledge Base for storing and retrieving market information."""
    
    def __init__(self, deployment: Dict[str, Any]):
        self.storage_provider = StorageClient(deployment.node)
        self.llm_config = deployment.config.llm_config
        self.chunker = SemanticChunker(
            min_size=512,
            max_size=1024
        )
        self.embedder = OllamaEmbedder(
            model=self.llm_config.model,
            url=self.llm_config.api_base
        )
        
        # Set up storage options
        storage_config = deployment.config.storage_config
        self.storage_type = storage_config.storage_type
        self.table_name = storage_config.path
        self.knowledge_schema = storage_config.storage_schema
        self.chunks_table = f"{self.table_name}_chunks"
        
    async def initialize(self, *args, **kwargs) -> Dict[str, Any]:
        """Initialize the knowledge base"""
        try:
            chunks_schema = {
                "id": {"type": "INTEGER", "primary_key": True},
                "knowledge_id": {"type": "TEXT"},
                "text": {"type": "TEXT"},
                "embedding": {"type": "VECTOR", "dimension": 768},
                "start": {"type": "INTEGER"},
                "ends_at": {"type": "INTEGER"}
            }
            
            # Check if tables exist, create if not
            if not await self.table_exists(self.table_name):
                logger.info(f"Creating table: {self.table_name}")
                # Create knowledge table
                create_request = CreateStorageRequest(
                    storage_type=self.storage_type,
                    path=self.table_name,
                    schema=self.knowledge_schema
                )
                await self.storage_provider.execute(create_request)
                
            if not await self.table_exists(self.chunks_table):
                logger.info(f"Creating chunks table: {self.chunks_table}")
                # Create chunks table
                create_request = CreateStorageRequest(
                    storage_type=self.storage_type,
                    path=self.chunks_table,
                    schema=chunks_schema
                )
                await self.storage_provider.execute(create_request)
                
            return {"status": "success", "message": "Knowledge base initialized"}
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {str(e)}")
            return {"status": "error", "message": str(e)}
            
    async def table_exists(self, table_name: str) -> bool:
        """Check if a table exists"""
        try:
            list_request = ListStorageRequest(
                storage_type=self.storage_type,
                path=table_name
            )
            await self.storage_provider.execute(list_request)
            return True
        except Exception:
            return False

    async def ingest_knowledge(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """ Process and store a document in the knowledge base with semantic chunking """
        try:
            await self.initialize()
            content = input_data.get("content", input_data.get("text", ""))
            
            knowledge_data = {
                "knowledge_id": str(uuid.uuid4()),
                "text": content,
                "metadata": input_data.get("metadata", {}),
                "source": input_data.get("metadata", {}).get("source", "unknown"),
                "timestamp": datetime.now(pytz.UTC).isoformat()
            }

            create_request = CreateStorageRequest(
                storage_type=self.storage_type,
                path=self.table_name,
                data={"data": knowledge_data}
            )

            knowledge_result = await self.storage_provider.execute(create_request)
            
            if not knowledge_result.data:
                return {"status": "error", "message": "Failed to create knowledge entry"}
                
            # Get the knowledge ID to link with chunks
            knowledge_id = knowledge_data["knowledge_id"]
            
            # Create semantic chunks
            chunks = self.chunker.chunk(content)
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = await self.embedder.embed_texts(chunk_texts)
            
            chunk_results = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_data = {
                    "knowledge_id": knowledge_id,
                    "text": chunk["text"],
                    "embedding": embedding,
                    "start": chunk["start"],
                    "ends_at": chunk["end"]
                }
                
                chunk_create_request = CreateStorageRequest(
                    storage_type=self.storage_type,
                    path=self.chunks_table,
                    data={"data": chunk_data}
                )
                
                chunk_result = await self.storage_provider.execute(chunk_create_request)
                chunk_results.append(chunk_result.data)
                
            return {
                "status": "success",
                "knowledge_id": knowledge_id,
                "chunks": len(chunk_results)
            }
            
        except Exception as e:
            logger.error(f"Error ingesting knowledge: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def search(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """ Search the knowledge base using semantic similarity """
        try:
            await self.initialize()
            
            query = input_data.get("query", "")
            top_k = input_data.get("top_k", 3)
            similarity_threshold = float(input_data.get("similarity_threshold", 0.3))
            
            if not query:
                return {"status": "error", "message": "Query is required"}
                
            query_embedding = await self.embedder.embed_text(query)
            if not query_embedding:
                return {"status": "error", "message": "Failed to generate embedding for query"}
            
            chunks_db_options = DatabaseReadOptions(
                filters={},
                ordering=[],
                vector_search={
                    "vector": query_embedding,
                    "column_name": "embedding",
                    "top_k": top_k,
                    "include_similarity": True,
                    "include_vectors": True
                }
            )
            
            chunks_read_request = ReadStorageRequest(
                storage_type=self.storage_type,
                path=self.chunks_table,
                options=chunks_db_options.model_dump()
            )
            
            chunk_results = await self.storage_provider.execute(chunks_read_request)
            
            if not chunk_results.data:
                return {"status": "success", "data": []}
                
            # Calculate proper similarity scores and filter results
            filtered_chunks = []
            for chunk in chunk_results.data:
                if "embedding" in chunk:
                    emb = chunk["embedding"]
                    if isinstance(emb, str):
                        try:
                            if emb.startswith('[') and emb.endswith(']'):
                                emb = json.loads(emb)
                            else:
                                emb = [float(x) for x in emb.split(',')]
                        except Exception:
                            continue
                    elif isinstance(emb, (list, tuple)):
                        try:
                            emb = [float(x) for x in emb]
                        except Exception:
                            continue
                    elif hasattr(emb, 'tolist'):
                        try:
                            emb = emb.tolist()
                        except Exception:
                            continue
                            
                    if emb is not None:
                        # Calculate similarity score
                        similarity_score = self._calculate_cosine_similarity(
                            query_embedding, 
                            emb,
                            query=query,
                            text=chunk["text"]
                        )
                        
                        # Only keep results above threshold
                        if similarity_score >= similarity_threshold:
                            chunk["similarity_score"] = similarity_score
                            filtered_chunks.append(chunk)
            
            if not filtered_chunks:
                return {"status": "success", "data": []}
            
            knowledge_ids = [chunk["knowledge_id"] for chunk in filtered_chunks]
            
            knowledge_read_request = ReadStorageRequest(
                storage_type=self.storage_type,
                path=self.table_name,
                options={"condition": {"knowledge_id": {"$in": knowledge_ids}}}
            )
            knowledge_results = await self.storage_provider.execute(knowledge_read_request)
            
            # Combine results with proper sorting by similarity
            combined_results = []
            for chunk in filtered_chunks:
                for knowledge in knowledge_results.data:
                    if chunk["knowledge_id"] == knowledge["knowledge_id"]:
                        combined_results.append({
                            "chunk": chunk["text"],
                            "chunk_start": chunk["start"],
                            "chunk_end": chunk["ends_at"],
                            "full_text": knowledge["text"],
                            "metadata": knowledge["metadata"],
                            "source": knowledge["source"],
                            "timestamp": knowledge["timestamp"],
                            "similarity_score": chunk["similarity_score"]
                        })
            
            combined_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            final_results = combined_results[:top_k]
            
            return {"status": "success", "data": final_results}
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {str(e)}")
            return {"status": "error", "message": str(e)}
            
    def _calculate_cosine_similarity(self, vec1, vec2, query=None, text=None):
        """Calculate similarity using both cosine similarity and keyword matching"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        try:
            try:
                vec1 = [float(x) for x in vec1]
                vec2 = [float(x) for x in vec2]
                
                dot_product = sum(a * b for a, b in zip(vec1, vec2))
                mag1 = math.sqrt(sum(a * a for a in vec1))
                mag2 = math.sqrt(sum(b * b for b in vec2))
                
                if mag1 < 1e-10 or mag2 < 1e-10:
                    return 0.0
                    
                cosine_sim = dot_product / (mag1 * mag2)
                cosine_sim = max(-1.0, min(1.0, cosine_sim))
                
            except (TypeError, ValueError):
                return 0.0
            
            if cosine_sim <= 0:
                semantic_score = 0.0
            else:
                shifted = (cosine_sim - 0.7) * 5
                semantic_score = 1 / (1 + math.exp(-shifted))
            
            keyword_boost = 0.0
            if query and text:
                query_words = {w.strip('?.,!') for w in query.lower().split() if len(w.strip('?.,!')) > 3}
                text_words = {w.strip('?.,!') for w in text.lower().split() if len(w.strip('?.,!')) > 3}
                
                common_words = query_words.intersection(text_words)
                if common_words and query_words:
                    overlap_ratio = len(common_words) / len(query_words)
                    if overlap_ratio >= 0.5:
                        keyword_boost = overlap_ratio * 0.3
            
            final_score = semantic_score * 0.8 + keyword_boost
            return min(1.0, final_score)
            
        except Exception:
            return 0.0

    async def get_by_id(self, knowledge_id: str) -> Dict[str, Any]:
        """ Retrieve a specific knowledge entry by ID """
        try:
            read_request = ReadStorageRequest(
                storage_type=self.storage_type,
                path=self.table_name,
                options={"condition": {"id": knowledge_id}}
            )
            result = await self.storage_provider.execute(read_request)
            return {"status": "success", "data": result}
        except Exception as e:
            logger.error(f"Error retrieving knowledge by ID: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def clear(self, *args, **kwargs) -> Dict[str, Any]:
        """ Clear all data from both tables """
        try:
            # Clear main knowledge table
            delete_request = DeleteStorageRequest(
                storage_type=self.storage_type,
                path=self.table_name,
                options={}
            )
            await self.storage_provider.execute(delete_request)
            
            chunks_delete_request = DeleteStorageRequest(
                storage_type=self.storage_type,
                path=self.chunks_table,
                options={}
            )
            await self.storage_provider.execute(chunks_delete_request)
            
            return {"status": "success", "message": "Knowledge base cleared"}
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {str(e)}")
            return {"status": "error", "message": str(e)}

async def run(module_run: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
    """ Run the Market Knowledge Base deployment """
    try:
        module_run = KBRunInput(**module_run)
        module_run.inputs = InputSchema(**module_run.inputs)
        market_kb = MarketKB(module_run.deployment)

        method = getattr(market_kb, module_run.inputs.func_name, None)
        if not method:
            raise ValueError(f"Invalid function name: {module_run.inputs.func_name}")

        result = await method(module_run.inputs.func_input_data)
        return {"status": "success", "results": [json.dumps(result)]}
    except Exception as e:
        logger.error(f"Error in run: {str(e)}")
        return {
            "status": "error",
            "error": True,
            "error_message": f"Error in run: {str(e)}"
        }

if __name__ == "__main__":
    import asyncio
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment
    from tests import test_search, test_ingest
    import os

    async def main():
        """Example of how to use the Market KB module"""
        if not os.getenv("NODE_URL"):
            print("Please set NODE_URL environment variable to run this script directly")
            return

        # First ingest some test data
        # print("\nIngesting test data...")
        # await test_ingest()
        
        # Then try searching
        print("\nTesting search...")
        await test_search("What is Lorem Ipsum?", top_k=2)

    asyncio.run(main())