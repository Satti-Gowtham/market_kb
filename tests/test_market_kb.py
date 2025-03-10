import asyncio
import os
import json
from naptha_sdk.client.naptha import Naptha
from naptha_sdk.configs import setup_module_deployment
from naptha_sdk.user import sign_consumer_id
from market_kb.run import run

naptha = Naptha()

async def test_ingest():
    """Test the knowledge ingestion functionality"""
    deployment = await setup_module_deployment(
        "kb",
        "market_kb/configs/deployment.json",
        node_url=os.getenv("NODE_URL")
    )

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
    return result

async def test_search(query: str = "", top_k: int = 2):
    """Test the search functionality with custom query and top_k"""
    deployment = await setup_module_deployment(
        "kb",
        "market_kb/configs/deployment.json",
        node_url=os.getenv("NODE_URL")
    )

    search_run = {
        "inputs": {
            "func_name": "search",
            "func_input_data": {"query": query, "top_k": top_k}
        },
        "deployment": deployment,
        "consumer_id": naptha.user.id,
        "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY"))
    }

    response = await run(search_run)
    if response["status"] == "success":
        for result in response["results"]:
            for item in json.loads(result)['data']:
                print("Chunk:", item["chunk"])
                print("From document:", item["full_text"][:100] + "...")
                print("Source:", item["source"])
                print("-"*100)
    else:
        print("Search Error:", response["message"])
    return response

async def test_clear():
    """Test clearing the knowledge base"""
    deployment = await setup_module_deployment(
        "kb",
        "market_kb/configs/deployment.json",
        node_url=os.getenv("NODE_URL")
    )

    clear_run = {
        "inputs": {
            "func_name": "clear",
            "func_input_data": None
        },
        "deployment": deployment,
        "consumer_id": naptha.user.id,
        "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY"))
    }

    result = await run(clear_run)
    print("Clear Result:", result)
    return result