"""
Video generation core module
Contains the main orchestration logic for video generation across different providers
"""

import os
import traceback
from typing import List, cast, Optional, Any
from models.config_model import ModelInfo
from ..video_providers.video_base_provider import get_default_provider, VideoProviderBase
# Import all providers to ensure automatic registration (don't delete these imports)
from ..video_providers.volces_provider import VolcesVideoProvider  # type: ignore
from ..video_providers.cfgpu_provider import CfgpuVideoProvider  # type: ignore
from .video_canvas_utils import (
    send_video_start_notification,
    send_video_error_notification,
    process_video_result,
)
from services.config_service import FILES_DIR
from ..video_generation_utils import get_image_base64, get_video_base64

_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.3gp'}


def _resolve_media_url(ref: str) -> str:
    """
    Convert a file_id or filename to a base64 data URL.
    - Images: use get_image_base64
    - Videos (vi_* or video extensions): use get_video_base64
    - http/https/data URLs: return as-is
    """
    if ref.startswith(('http://', 'https://', 'data:')):
        return ref
    ref_stem = os.path.splitext(ref)[0]
    for fname in os.listdir(FILES_DIR):
        if fname == ref or os.path.splitext(fname)[0] == ref_stem:
            ext = os.path.splitext(fname)[1].lower()
            if ext in _VIDEO_EXTENSIONS:
                return get_video_base64(fname)
            else:
                return get_image_base64(fname)
    return ref


async def generate_video_with_provider(
    prompt: str,
    resolution: str,
    duration: int,
    aspect_ratio: str,
    model: str,
    tool_call_id: str,
    config: Any,
    input_images: Optional[list[str]] = None,
    input_videos: Optional[list[str]] = None,
    camera_fixed: bool = True,
    **kwargs: Any
) -> str:
    """
    Universal video generation function supporting different models and providers

    Args:
        prompt: Video generation prompt
        resolution: Video resolution (480p, 1080p)
        duration: Video duration in seconds (5, 10)
        aspect_ratio: Video aspect ratio (1:1, 16:9, 4:3, 21:9)
        model: Model identifier (e.g., 'doubao-seedance-1-0-pro')
        tool_call_id: Tool call ID
        config: Context runtime configuration containing canvas_id, session_id, model_info, injected by langgraph
        input_images: Optional input reference images list
        input_videos: Optional input reference videos list
        camera_fixed: Whether to keep camera fixed

    Returns:
        str: Generation result message
    """
    model_name = model.split(
        # Some model names contain "/", like "openai/gpt-image-1", need to handle
        '/')[-1]
    print(f'🛠️ Video Generation {model_name} tool_call_id', tool_call_id)
    ctx = config.get('configurable', {})
    canvas_id = ctx.get('canvas_id', '')
    session_id = ctx.get('session_id', '')
    print(f'🛠️ canvas_id {canvas_id} session_id {session_id}')

    # Inject the tool call id into the context
    ctx['tool_call_id'] = tool_call_id

    try:
        # Determine provider selection
        model_info_list: List[ModelInfo] = cast(
            List[ModelInfo], ctx.get('model_info', {}).get(model_name, []))

        if model_info_list == []:
            # video registed as tool
            model_info_list: List[ModelInfo] = cast(
                List[ModelInfo], ctx.get('tool_list', {}))

        # Use get_default_provider which already handles Jaaz prioritization
        provider_name = get_default_provider(model_info_list)

        print(f"🎥 Using provider: {provider_name} for {model_name}")

        # Create provider instance
        provider_instance = VideoProviderBase.create_provider(provider_name)

        # Send start notification
        await send_video_start_notification(
            session_id,
            f"Starting video generation using {model_name} via {provider_name}..."
        )

        # Convert file_ids to base64 data URLs for external providers
        processed_input_images = [_resolve_media_url(r) for r in input_images] if input_images else None
        processed_input_videos = [_resolve_media_url(r) for r in input_videos] if input_videos else None

        # Generate video using the selected provider
        video_url = await provider_instance.generate(
            prompt=prompt,
            model=model,
            resolution=resolution,
            duration=duration,
            aspect_ratio=aspect_ratio,
            input_images=processed_input_images,
            input_videos=processed_input_videos,
            camera_fixed=camera_fixed,
            **kwargs
        )

        # Process video result (save, update canvas, notify)
        return await process_video_result(
            video_url=video_url,
            session_id=session_id,
            canvas_id=canvas_id,
            provider_name=f"{model_name} ({provider_name})"
        )

    except Exception as e:
        error_message = str(e)
        print(f"🎥 Error generating video with {model_name}: {error_message}")
        traceback.print_exc()

        # Send error notification
        await send_video_error_notification(session_id, error_message)

        # Re-raise the exception for proper error handling
        raise Exception(
            f"{model_name} video generation failed: {error_message}")
