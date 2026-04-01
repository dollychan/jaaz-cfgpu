from pydantic import BaseModel
from typing import Literal, TypedDict

class LLMConfig(BaseModel):
    model: str
    base_url: str
    api_key: str
    max_tokens: int
    temperature: float

class ConfigUpdate(BaseModel):
    llm: LLMConfig

class ModelInfo(TypedDict, total=False):
    provider: str
    model: str # For tool type, it is the function name
    url: str
    type: Literal['text', 'image', 'tool', 'video']
    api_key: str  # Optional: overrides config_service lookup when set
