"""
Video generation core module
Contains the main orchestration logic for video generation across different providers
"""

import asyncio
import base64
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
from ..video_generation_utils import get_image_base64
from utils.http_client import HttpClient

_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.3gp'}
_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.aac', '.m4a', '.ogg', '.flac', '.opus'}

# Public base URL of this jaaz server, used so external APIs can fetch local video files.
# Set JAAZ_SERVER_URL env var to the publicly accessible address, e.g. http://1.2.3.4:57988
# Read at call time (not module import time) so changes to the env var after startup take effect.
def _get_server_base_url() -> str:
    return os.environ.get("JAAZ_SERVER_URL", "http://127.0.0.1:57988").rstrip("/")


async def _resolve_image_url(ref: str) -> str:
    """
    Resolve an image reference to a base64 data URL.
    - data: URLs → returned as-is
    - http/https URLs → downloaded and converted to base64
    - local file_id/filename → read from FILES_DIR and converted to base64
    """
    if ref.startswith('data:'):
        return ref
    if ref.startswith(('http://', 'https://')):
        try:
            async with HttpClient.create_aiohttp() as session:
                async with session.get(ref) as response:
                    data = await response.read()
                    content_type = response.headers.get('Content-Type', 'image/jpeg').split(';')[0]
                    b64 = base64.b64encode(data).decode('utf-8')
                    print(f"🖼️ Downloaded remote image '{ref[:60]}...' → base64 ({len(data)} bytes)")
                    return f"data:{content_type};base64,{b64}"
        except Exception as e:
            print(f"⚠️ Failed to download image '{ref}': {e}, passing URL as-is")
            return ref
    ref_stem = os.path.splitext(ref)[0]
    for fname in os.listdir(FILES_DIR):
        if fname == ref or os.path.splitext(fname)[0] == ref_stem:
            return get_image_base64(fname)
    return ref


def _resolve_local_to_server_url(
    ref: str,
    allowed_extensions: Optional[set] = None,
    label: str = "media",
) -> str:
    """
    Convert a local file_id/filename to a server-hosted URL.

    External APIs (cfgpu, etc.) require real web URLs for video/audio.
    - http/https/data: URLs are returned unchanged.
    - Local refs are matched by exact filename or stem, then filtered by
      allowed_extensions when provided (None = accept any extension).
    - Falls back to the original ref with a warning if not found.
    """
    if ref.startswith(('http://', 'https://', 'data:')):
        return ref
    ref_stem = os.path.splitext(ref)[0]
    for fname in os.listdir(FILES_DIR):
        if fname == ref or os.path.splitext(fname)[0] == ref_stem:
            ext = os.path.splitext(fname)[1].lower()
            if allowed_extensions is None or ext in allowed_extensions:
                server_url = f"{_get_server_base_url()}/api/file/{fname}"
                print(f"🔗 Resolved local {label} '{ref}' → {server_url}")
                return server_url
    print(f"⚠️ Local {label} '{ref}' not found, passing as-is")
    return ref


def _resolve_video_url(ref: str) -> str:
    return _resolve_local_to_server_url(ref, label="video")


def _resolve_audio_url(ref: str) -> str:
    return _resolve_local_to_server_url(ref, _AUDIO_EXTENSIONS, label="audio")


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
    input_audios: Optional[list[str]] = None,
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

        # Images → base64 data URL (download remote URLs if needed)
        # Videos / Audios → server-hosted web URL (external APIs require real URLs)
        processed_input_images = list(await asyncio.gather(*[_resolve_image_url(r) for r in input_images])) if input_images else None
        processed_input_videos = [_resolve_video_url(r) for r in input_videos] if input_videos else None
        processed_input_audios = [_resolve_audio_url(r) for r in input_audios] if input_audios else None

        # Generate video using the selected provider
        video_url = await provider_instance.generate(
            prompt=prompt,
            model=model,
            resolution=resolution,
            duration=duration,
            aspect_ratio=aspect_ratio,
            input_images=processed_input_images,
            input_videos=processed_input_videos,
            input_audios=processed_input_audios,
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
