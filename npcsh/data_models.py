from pydantic import BaseModel
from typing import List, Dict


class NPC(BaseModel):
    name: str
    primary_directive: str
    model: str
    provider: str
    api_url: str
    tools: List[str]
    use_default_tools: bool


class Tool(BaseModel):
    tool_name: str
    tool_description: str
    steps: List[Dict[str, str]]
