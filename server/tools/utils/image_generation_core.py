"""
Image generation core module
Contains the main orchestration logic for image generation across different providers
"""

import os
from typing import Optional, Dict, Any
from tools.utils.image_utils import process_input_image
from ..image_providers.image_base_provider import ImageProviderBase

# 导入所有提供商以确保自动注册 (不要删除这些导入)
from ..image_providers.jaaz_provider import JaazImageProvider
from ..image_providers.openai_provider import OpenAIImageProvider
from ..image_providers.replicate_provider import ReplicateImageProvider
from ..image_providers.volces_provider import VolcesProvider
from ..image_providers.wavespeed_provider import WavespeedProvider
from ..image_providers.cfgpu_provider import CfgpuImageProvider

# from ..image_providers.comfyui_provider import ComfyUIProvider
from .image_canvas_utils import (
    save_image_to_canvas,
)
import time

IMAGE_PROVIDERS: dict[str, ImageProviderBase] = {
    "jaaz": JaazImageProvider(),
    "openai": OpenAIImageProvider(),
    "replicate": ReplicateImageProvider(),
    "volces": VolcesProvider(),
    "wavespeed": WavespeedProvider(),
    "cfgpu": CfgpuImageProvider(),
}


async def generate_image_with_provider(
    canvas_id: str,
    session_id: str,
    provider: str,
    model: str,
    # image generator args
    prompt: str,
    aspect_ratio: str = "1:1",
    input_images: Optional[list[str]] = None,
    **kwargs: Any,
) -> str:
    """
    通用图像生成函数，支持不同的模型和提供商

    Args:
        prompt: 图像生成提示词
        aspect_ratio: 图像长宽比
        model_name: 内部模型名称 (如 'gpt-image-1', 'imagen-4')
        model: 模型标识符 (如 'openai/gpt-image-1', 'google/imagen-4')
        tool_call_id: 工具调用ID
        config: 上下文运行配置，包含canvas_id，session_id，model_info，由langgraph注入
        input_images: 可选的输入参考图像列表

    Returns:
        str: 生成结果消息
    """

    provider_instance = IMAGE_PROVIDERS.get(provider)
    if not provider_instance:
        raise ValueError(f"Unknown provider: {provider}")

    # Process input images for the provider
    processed_input_images: list[str] | None = None
    if input_images:
        processed_input_images = []
        for image_path in input_images:
            processed_image = process_input_image(image_path)
            if processed_image:
                processed_input_images.append(processed_image)

        print(f"Using {len(processed_input_images)} input images for generation")

    # Prepare metadata with all generation parameters
    metadata: Dict[str, Any] = {
        "prompt": prompt,
        "model": model,
        "provider": provider,
        "aspect_ratio": aspect_ratio,
        "input_images": input_images or [],
    }

    try:
        # Generate image using the selected provider
        mime_type, width, height, filename = await provider_instance.generate(
            prompt=prompt,
            model=model,
            aspect_ratio=aspect_ratio,
            input_images=processed_input_images,
            metadata=metadata,
            **kwargs,
        )
    except Exception as e:
        error_message = str(e)
        print(f"🖼️ Error generating image with {provider}/{model}: {error_message}")
        # Return error as tool result so it appears in the tool call box.
        # Do NOT raise — that would bypass LangGraph's tool result handling
        # and potentially trigger retries or a global error toast.
        return f"Image generation failed: {error_message}"

    # Save image to canvas
    image_url = await save_image_to_canvas(
        session_id, canvas_id, filename, mime_type, width, height
    )

    server_base = os.environ.get("JAAZ_SERVER_URL", f"http://127.0.0.1:{os.environ.get('DEFAULT_PORT', '57988')}").rstrip("/")
    return f"image generated successfully ![image_id: {filename}]({server_base}{image_url})"
