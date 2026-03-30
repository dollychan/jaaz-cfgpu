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
        size: str = "2K",
        **kwargs: Any,
    ) -> Tuple[str, int, int, str]:
        """
        Generate image using CFGPU Doubao Seedream API.

        Args:
            prompt: Image generation prompt
            model: Model name (e.g. doubao-seedream-5-0-260128)
            aspect_ratio: Ignored — use `size` to control output dimensions
            input_images: Not supported by this model
            size: Output size preset (e.g. "1K", "2K", "4K"). Default "2K"

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
            payload = {
                "model": model,
                "prompt": prompt,
                "sequential_image_generation": "disabled",
                "response_format": "url",
                "size": size,
                "stream": False,
                "watermark": False,
            }

            print(f"🖼️ CFGPU image generation: model={model}, size={size}")

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
