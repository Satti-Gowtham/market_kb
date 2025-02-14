import pytz
from typing import Dict, Any
from datetime import datetime
from uuid import UUID

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

from market_kb.schemas import InputSchema, RetrievedMemory
from market_kb.utils.chunker import SemanticChunker
from market_kb.utils.embeddings import OllamaEmbedder

logger = get_logger(__name__)

class MarketKB:
    def __init__(self, deployment: Dict[str, Any]):
        self.deployment = deployment
        self.config = self.deployment.config
        self.storage_provider = StorageClient(self.deployment.node)
        self.storage_type = self.config.storage_config.storage_type
        self.table_name = self.config.storage_config.path
        self.chunks_table = f"{self.table_name}_chunks"
        self.knowledge_schema = self.config.storage_config.storage_schema
        self.chunks_schema = {
            "id": {"type": "INTEGER", "primary_key": True},
            "knowledge_id": {"type": "INTEGER"},
            "text": {"type": "TEXT"},
            "embedding": {"type": "VECTOR", "dimension": 768},
            "start": {"type": "INTEGER"},
            "ends_at": {"type": "INTEGER"}
        }
        self.chunker = SemanticChunker(min_size=512, max_size=1024)
        
        # Initialize embeddings
        llm_config = self.config.llm_config
        self.embeddings = OllamaEmbedder(
            model=llm_config.model,
            url=llm_config.api_base
        )

    async def initialize(self, *args, **kwargs):
        """Initialize knowledge base and chunks tables"""
        logger.info(f"Initializing tables {self.table_name} and {self.chunks_table}")
        
        try:
            # Create main knowledge table
            if not await self.table_exists(self.table_name):
                create_request = CreateStorageRequest(
                    storage_type=self.storage_type,
                    path=self.table_name,
                    data={"schema": self.knowledge_schema}
                )
                await self.storage_provider.execute(create_request)
                logger.info(f"Created table {self.table_name}")
            else:
                logger.info(f"Table {self.table_name} already exists")

            # Create chunks table
            if not await self.table_exists(self.chunks_table):
                create_request = CreateStorageRequest(
                    storage_type=self.storage_type,
                    path=self.chunks_table,
                    data={"schema": self.chunks_schema}
                )
                await self.storage_provider.execute(create_request)
                logger.info(f"Created table {self.chunks_table}")
            else:
                logger.info(f"Table {self.chunks_table} already exists")

            return {"status": "success", "message": "Tables initialized successfully"}
        except Exception as e:
            logger.error(f"Error initializing tables: {str(e)}")
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
        """Process and store a document in the knowledge base with semantic chunking"""
        try:
            await self.initialize()
            # First store the full document
            knowledge_data = {
                "text": input_data["text"],
                "metadata": input_data.get("metadata", {}),
                "source": input_data.get("metadata", {}).get("source"),
                "timestamp": datetime.now(pytz.UTC).isoformat()
            }

            create_request = CreateStorageRequest(
                storage_type=self.storage_type,
                path=self.table_name,
                data={"data": knowledge_data}
            )

            knowledge_result = await self.storage_provider.execute(create_request)
            knowledge_id = knowledge_result.data["data"]["id"]

            # Then store the chunks with embeddings
            chunks = self.chunker.chunk(input_data["text"])
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = await self.embeddings.embed_texts(chunk_texts)

            logger.info(f"Embedding {len(embeddings)} chunks")

            for chunk, embedding in zip(chunks, embeddings):
                chunk_data = {
                    "knowledge_id": knowledge_id,
                    "text": chunk["text"],
                    "start": chunk["start"],
                    "ends_at": chunk["end"],
                    "embedding": embedding
                }

                create_request = CreateStorageRequest(
                    storage_type=self.storage_type,
                    path=self.chunks_table,
                    data={"data": chunk_data}
                )
                await self.storage_provider.execute(create_request)

            return UUID(knowledge_id)
        except Exception as e:
            logger.error(f"Error ingesting knowledge: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def search(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Search the knowledge base using semantic similarity"""
        try:
            query = input_data.get("query")
            top_k = input_data.get("top_k", 2)
            
            # Get embedding for query
            query_embedding = await self.embeddings.embed_text(query)
            
            # Search in chunks table using vector similarity
            chunks_db_options = DatabaseReadOptions(
                query_vector=query_embedding,
                vector_col="embedding",
                top_k=top_k,
                include_similarity=True
            )
            
            chunks_read_request = ReadStorageRequest(
                storage_type=self.storage_type,
                path=self.chunks_table,
                options=chunks_db_options.model_dump()
            )
            chunk_results = await self.storage_provider.execute(chunks_read_request)

            if not chunk_results.data:
                return {"status": "success", "data": []}

            # Get the corresponding knowledge entries
            knowledge_ids = [chunk["knowledge_id"] for chunk in chunk_results.data]
            
            # Get full knowledge entries
            knowledge_read_request = ReadStorageRequest(
                storage_type=self.storage_type,
                path=self.table_name,
                options={"condition": {"id": {"$in": knowledge_ids}}}
            )
            knowledge_results = await self.storage_provider.execute(knowledge_read_request)

            # Combine results
            combined_results = []
            for chunk in chunk_results.data:
                for knowledge in knowledge_results.data:
                    if chunk["knowledge_id"] == knowledge["id"]:
                        combined_results.append({
                            "chunk": chunk["text"],
                            "chunk_start": chunk["start"],
                            "chunk_end": chunk["ends_at"],
                            "full_text": knowledge["text"],
                            "metadata": knowledge["metadata"],
                            "source": knowledge["source"],
                            "timestamp": knowledge["timestamp"]
                        })

            return [RetrievedMemory(**result) for result in combined_results[:top_k]]
        except Exception as e:
            logger.error(f"Error searching knowledge base: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def get_by_id(self, knowledge_id: str) -> Dict[str, Any]:
        """Retrieve a specific knowledge entry by ID"""
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
        """Clear all data from both tables"""
        try:
            # Clear main knowledge table
            delete_request = DeleteStorageRequest(
                storage_type=self.storage_type,
                path=self.table_name,
                condition={}
            )
            await self.storage_provider.execute(delete_request)

            # Clear chunks table
            delete_request = DeleteStorageRequest(
                storage_type=self.storage_type,
                path=self.chunks_table,
                condition={}
            )
            await self.storage_provider.execute(delete_request)

            return {"status": "success", "message": "Cleared all tables"}
        except Exception as e:
            logger.error(f"Error clearing tables: {str(e)}")
            return {"status": "error", "message": str(e)}

async def run(module_run: Dict[str, Any], *args, **kwargs):
    """Run the Market Knowledge Base deployment"""
    module_run = KBRunInput(**module_run)
    module_run.inputs = InputSchema(**module_run.inputs)
    market_kb = MarketKB(module_run.deployment)

    method = getattr(market_kb, module_run.inputs.func_name, None)
    if not method:
        raise ValueError(f"Invalid function name: {module_run.inputs.func_name}")

    return await method(module_run.inputs.func_input_data)

if __name__ == "__main__":
    import asyncio
    import os
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment

    naptha = Naptha()
    
    async def test_kb():
        deployment = await setup_module_deployment(
            "kb",
            "market_kb/configs/deployment.json",
            node_url=os.getenv("NODE_URL")
        )

        # Test ingestion
        ingest_run = {
            "inputs": {
                "func_name": "ingest_knowledge",
                "func_input_data": {
                    "text": '''
Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, "Lorem ipsum dolor sit amet..", comes from a line in section 1.10.32.

The standard chunk of Lorem Ipsum used since the 1500s is reproduced below for those interested. Sections 1.10.32 and 1.10.33 from "de Finibus Bonorum et Malorum" by Cicero are also reproduced in their exact original form, accompanied by English versions from the 1914 translation by H. Rackham.''',
                    "metadata": {"source": "test"}
                }
            },
            "deployment": deployment,
            "consumer_id": naptha.user.id,
            "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY"))
        }
        
        result = await run(ingest_run)
        print("Ingest Result:", result)

        # Test search
        search_run = {
            "inputs": {
                "func_name": "search",
                "func_input_data": {"query": "What is Lorem Ipsum?", "top_k": 2}
            },
            "deployment": deployment,
            "consumer_id": naptha.user.id,
            "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY"))
        }

        results = await run(search_run)
        if results["status"] == "success":
            for result in results["data"]:
                print("Chunk:", result["chunk"])
                print("From document:", result["full_text"][:100] + "...")
                print("Source:", result["source"])
                print("-"*100)
        else:
            print("Search Error:", results["message"])

        # Test clear
        # clear_run = {
        #     "inputs": {
        #         "func_name": "clear",
        #         "func_input_data": None
        #     },
        #     "deployment": deployment,
        #     "consumer_id": naptha.user.id,
        #     "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY"))
        # }

        # result = await run(clear_run)
        # print("Clear Result:", result)

    asyncio.run(test_kb())