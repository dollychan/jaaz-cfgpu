from typing import Annotated, List, Optional
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
    input_images: Optional[List[str]] = Field(
        default=None,
        description="Optional. List of reference image URLs or file IDs. Usage depends on image_role."
    )
    image_role: str = Field(
        default="auto",
        description=(
            "Optional. How input images are used. Ignored when input_images is not set. "
            "'auto' (default): infer from count — 1 image → first_frame, 2 images → first_last_frame, 3–9 images → reference_image; "
            "automatically switches to reference_image when input_videos or input_audios are also provided. "
            "'first_frame': use the single image as the video's starting frame (image-to-video, no input_videos/input_audios allowed). "
            "'first_last_frame': use exactly 2 images as starting and ending frames (no input_videos/input_audios allowed). "
            "'reference_image': use 1–9 images as style/content references; compatible with input_videos and input_audios (multimodal reference-to-video)."
        )
    )
    input_videos: Optional[List[str]] = Field(
        default=None,
        description="Optional. List of reference video URLs or file IDs for video-to-video or video extension."
    )
    input_audios: Optional[List[str]] = Field(
        default=None,
        description="Optional. List of reference audio URLs or file IDs. MUST be used together with input_images or input_videos — audio cannot be the only reference input."
    )
    generate_audio: bool = Field(
        default=True,
        description="Optional. Whether the output video should contain synchronized audio (voice, sound effects, background music). Default is True. Set to False for silent video output."
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


@tool("generate_video_by_cfgpu_seedance_2_0_fast",
      description="Generate videos quickly using Wan-Video-Fast model via CFGPU provider. Faster generation, supports text-to-video, image-to-video (first frame / bookend frames / multimodal reference), video-to-video, and audio-guided generation.",
      args_schema=GenerateByCfgpuSeedance2_0FastInputSchema)
async def generate_video_by_cfgpu_seedance_2_0_fast(
    prompt: str,
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    duration: int = 5,
    aspect_ratio: str = "16:9",
    input_images: Optional[List[str]] = None,
    image_role: str = "auto",
    input_videos: Optional[List[str]] = None,
    input_audios: Optional[List[str]] = None,
    generate_audio: bool = True,
) -> str:
    return await generate_video_with_provider(
        prompt=prompt,
        resolution="480p",
        duration=duration,
        aspect_ratio=aspect_ratio,
        model="wan-video-fast",
        tool_call_id=tool_call_id,
        config=config,
        input_images=input_images,
        input_videos=input_videos,
        input_audios=input_audios,
        image_role=image_role,
        generate_audio=generate_audio,
    )


__all__ = ["generate_video_by_cfgpu_seedance_2_0_fast"]
