from pydantic import BaseModel
from typing import Literal, Optional, Dict, Any
# from naptha_sdk.schemas import KBConfig

class InputSchema(BaseModel):
    func_name: Literal[
        "initialize", 
        "run_query", 
        "add_data", 
        "delete_table", 
        "delete_row", 
        "list_rows",
        "ingest_knowledge",
        "search",
        "get_by_id",
        "clear"
    ]
    func_input_data: Optional[Dict[str, Any]] = None

class OutputSchema(BaseModel):
    success: bool
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None