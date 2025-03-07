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
    description: str
    steps: List[Dict[str, str]]


class ToolStep(BaseModel):
    engine: str
    code: str


class Context(BaseModel):
    databases: List[str]
    files: List[str]
    vars: List[Dict[str, str]]


class Pipeline(BaseModel):
    steps: List[Dict[str, str]]


class PipelineStep(BaseModel):
    tool: str
    args: List[str]
    model: str
    provider: str
    task: str
    npc: str


class Fabrication(BaseModel):
    spell: str
