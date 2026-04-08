import os
import traceback
from typing import Optional, Any, Tuple

from .image_base_provider import ImageProviderBase
from ..utils.image_utils import get_image_info_and_save, generate_image_id
from services.config_service import FILES_DIR, config_service
from utils.http_client import HttpClient


class CfgpuImageProvider(ImageProviderBase):
    """CFGPU image generation provider (Doubao Seedream models)"""

    def _get_config(self):
        config = config_service.app_config.get('cfgpu', {})
        api_key = config.get("api_key", "")
        if not api_key:
            raise ValueError("CFGPU API key is not configured")
        text_url = config.get("url", "https://www.cfgpu.com/userapi/v1/model/v1/").rstrip("/")
        base_url = text_url.removesuffix("/model/v1")
        return api_key, base_url

    async def generate(
        self,
        prompt: str,
        model: str,
        aspect_ratio: str = "1:1",
        input_images: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        size: str = "2k",
        **kwargs: Any,
    ) -> Tuple[str, int, int, str]:
        """
        Generate image using CFGPU Doubao Seedream API.

        Args:
            prompt: Image generation prompt
            model: Model name (e.g. doubao-seedream-5-0-260128)
            aspect_ratio: Ignored — use `size` to control output dimensions
            input_images: Optional reference images as base64 data URLs (data:image/<fmt>;base64,...).
                          Supported by doubao-seedream-4.0/4.5/5.0-lite; up to 14 images.
                          NOT supported by doubao-seedream-3.0-t2i.
            size: Output size preset. API accepts '2k', '3k', or 'WIDTHxHEIGHT'. Default "2k"

        Returns:
            Tuple[str, int, int, str]: (mime_type, width, height, filename)
        """
        try:
            api_key, base_url = self._get_config()
            url = f"{base_url}/images/generations"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            # API requires lowercase size: '2k', '3k', or 'WIDTHxHEIGHT'
            normalized_size = size.lower() if size else "2k"
            payload: dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "sequential_image_generation": "disabled",
                "response_format": "url",
                "size": normalized_size,
                "stream": False,
                "watermark": False,
            }

            # Pass reference images when provided.
            # Single image → string; multiple images → array (max 14 per API docs).
            if input_images:
                clamped = input_images[:14]
                payload["image"] = clamped[0] if len(clamped) == 1 else clamped

            print(f"🖼️ CFGPU image generation: model={model}, size={normalized_size}, images={len(input_images) if input_images else 0}")

            async with HttpClient.create_aiohttp() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        try:
                            error_data = await response.json()
                        except Exception:
                            error_data = await response.text()
                        raise Exception(f"CFGPU image generation failed ({response.status}): {error_data}")
                    result = await response.json()

            # OpenAI-compatible response: { "data": [{ "url": "..." }] }
            image_url = (
                (result.get("data") or [{}])[0].get("url")
                or result.get("url")
            )
            if not image_url:
                raise Exception(f"No image URL in CFGPU response: {result}")

            print(f"🖼️ CFGPU image URL: {image_url[:80]}...")

            image_id = generate_image_id()
            mime_type, width, height, extension = await get_image_info_and_save(
                image_url, os.path.join(FILES_DIR, f"{image_id}"), is_b64=False
            )
            filename = f"{image_id}.{extension}"
            return mime_type, width, height, filename

        except Exception as e:
            print(f"🖼️ Error generating image with CFGPU: {e}")
            traceback.print_exc()
            raise e
