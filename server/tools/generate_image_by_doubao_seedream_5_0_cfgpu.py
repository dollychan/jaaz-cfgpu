from typing import Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool, InjectedToolCallId  # type: ignore
from langchain_core.runnables import RunnableConfig
from tools.utils.image_generation_core import generate_image_with_provider


class GenerateImageByDoubaoSeedream5_0CfgpuInputSchema(BaseModel):
    prompt: str = Field(
        description="Required. The prompt for image generation. Describe what you want to see in the image."
    )
    size: str = Field(
        default="2K",
        description="Optional. Output image size. Allowed values: 1K, 2K, 4K. Default is 2K."
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


@tool("generate_image_by_doubao_seedream_5_0_cfgpu",
      description="Generate a high-quality image using Doubao Seedream 5.0 model via CFGPU. Latest and most capable Seedream model. Text-to-image only.",
      args_schema=GenerateImageByDoubaoSeedream5_0CfgpuInputSchema)
async def generate_image_by_doubao_seedream_5_0_cfgpu(
    prompt: str,
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    size: str = "2K",
) -> str:
    ctx = config.get('configurable', {})
    return await generate_image_with_provider(
        canvas_id=ctx.get('canvas_id', ''),
        session_id=ctx.get('session_id', ''),
        provider='cfgpu',
        model='doubao-seedream-5-0-260128',
        prompt=prompt,
        size=size,
    )


__all__ = ["generate_image_by_doubao_seedream_5_0_cfgpu"]
