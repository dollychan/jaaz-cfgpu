from typing import Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool, InjectedToolCallId  # type: ignore
from langchain_core.runnables import RunnableConfig
from tools.video_generation.video_generation_core import generate_video_with_provider


class GenerateByCfgpuSeedance2_0FastInputSchema(BaseModel):
    prompt: str = Field(
        description="Required. The prompt for video generation. Describe what you want to see in the video."
    )
    duration: int = Field(
        default=5,
        description="Optional. The duration of the video in seconds. Default is 5."
    )
    aspect_ratio: str = Field(
        default="16:9",
        description="Optional. The aspect ratio of the video. Allowed values: 16:9, 9:16, 1:1, 4:3, 3:4, 21:9"
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


@tool("generate_video_by_cfgpu_seedance_2_0_fast",
      description="Generate videos quickly using Wan-Video-Fast model via CFGPU provider. Faster than Wan-Video, supports text-to-video generation.",
      args_schema=GenerateByCfgpuSeedance2_0FastInputSchema)
async def generate_video_by_cfgpu_seedance_2_0_fast(
    prompt: str,
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    duration: int = 5,
    aspect_ratio: str = "16:9",
) -> str:
    return await generate_video_with_provider(
        prompt=prompt,
        resolution="480p",
        duration=duration,
        aspect_ratio=aspect_ratio,
        model="wan-video-fast",
        tool_call_id=tool_call_id,
        config=config,
    )


__all__ = ["generate_video_by_cfgpu_seedance_2_0_fast"]
