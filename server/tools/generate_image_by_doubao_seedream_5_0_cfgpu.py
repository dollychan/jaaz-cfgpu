from typing import Annotated, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool, InjectedToolCallId  # type: ignore
from langchain_core.runnables import RunnableConfig
from tools.utils.image_generation_core import generate_image_with_provider
from tools.utils.schema_validators import OptionalStringList


class GenerateImageByDoubaoSeedream5_0CfgpuInputSchema(BaseModel):
    prompt: str = Field(
        description="Required. The prompt for image generation. Describe what you want to see in the image."
    )
    size: str = Field(
        default="2K",
        description="Optional. Output image size. Allowed values: 1K, 2K, 4K. Default is 2K."
    )
    input_images: OptionalStringList = Field(
        default=None,
        description="Optional. 1–14 reference image file_ids (e.g. ['im_abc123.png']). "
                    "Use for image-to-image generation: style transfer, character consistency, object editing, etc."
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


@tool("generate_image_by_doubao_seedream_5_0_cfgpu",
      description="Generate a high-quality image using Doubao Seedream 5.0 model via CFGPU. "
                  "Latest and most capable Seedream model. Supports text-to-image and image-to-image (1–14 reference images).",
      args_schema=GenerateImageByDoubaoSeedream5_0CfgpuInputSchema)
async def generate_image_by_doubao_seedream_5_0_cfgpu(
    prompt: str,
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    size: str = "2K",
    input_images: Optional[list[str]] = None,
) -> str:
    ctx = config.get('configurable', {})
    return await generate_image_with_provider(
        canvas_id=ctx.get('canvas_id', ''),
        session_id=ctx.get('session_id', ''),
        provider='cfgpu',
        model='doubao-seedream-5-0-260128',
        prompt=prompt,
        size=size,
        input_images=input_images,
    )


__all__ = ["generate_image_by_doubao_seedream_5_0_cfgpu"]
