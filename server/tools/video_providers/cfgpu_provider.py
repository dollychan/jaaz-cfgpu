import asyncio
import traceback
from typing import Optional, Dict, Any, List

from .video_base_provider import VideoProviderBase
from utils.http_client import HttpClient
from services.config_service import config_service


class CfgpuVideoProvider(VideoProviderBase, provider_name="cfgpu"):
    """CFGPU video generation provider implementation"""

    def __init__(self):
        config = config_service.app_config.get('cfgpu', {})
        self.api_key = config.get("api_key", "")
        text_url = config.get("url", "https://www.cfgpu.com/userapi/v1/model/v1/").rstrip("/")
        # Text URL is .../model/v1; video URL lives at .../video, so strip /model/v1
        self.base_url = text_url.removesuffix("/model/v1")

        if not self.api_key:
            raise ValueError("CFGPU API key is not configured")

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_request_payload(
        self,
        prompt: str,
        model: str,
        aspect_ratio: str = "16:9",
        duration: int = 5,
        input_image_data: Optional[List[str]] = None,
        input_video_data: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        if input_image_data:
            for img_url in input_image_data:
                content.append({"type": "image_url", "image_url": {"url": img_url}, "role": "reference_image"})

        if input_video_data:
            for vid_url in input_video_data:
                content.append({"type": "video_url", "video_url": {"url": vid_url}, "role": "reference_video"})

        return {
            "model": model,
            "content": content,
            "ratio": aspect_ratio,
            "duration": duration,
            "watermark": False,
        }

    async def _poll_task_status(self, task_id: str, model: str, headers: Dict[str, str]) -> str:
        """Poll task status until completion"""
        polling_url = f"{self.base_url}/video/tasks/{model}/{task_id}"
        status = "pending"

        async with HttpClient.create_aiohttp() as session:
            while status not in ("succeeded", "failed", "cancelled"):
                print(f"🎥 Polling CFGPU generation {task_id}, current status: {status} ...")
                await asyncio.sleep(5)

                async with session.get(polling_url, headers=headers) as poll_response:
                    poll_res = await poll_response.json()
                    status = poll_res.get("status", "pending")

                    if status == "succeeded":
                        content = poll_res.get("content") or {}
                        video_url = (
                            content.get("videoUrl")
                            or content.get("video_url")
                            or poll_res.get("video_url")
                            or poll_res.get("url")
                        )
                        if video_url and isinstance(video_url, str):
                            return video_url
                        raise Exception(f"No video URL found in successful response: {poll_res}")
                    elif status in ("failed", "cancelled"):
                        detail = poll_res.get("message", f"Task failed with status: {status}")
                        raise Exception(f"CFGPU video generation failed: {detail}")

        raise Exception(f"Task polling failed with final status: {status}")

    async def generate(
        self,
        prompt: str,
        model: str,
        resolution: str = "480p",
        duration: int = 5,
        aspect_ratio: str = "16:9",
        input_images: Optional[List[str]] = None,
        input_videos: Optional[List[str]] = None,
        camera_fixed: bool = True,
        **kwargs: Any
    ) -> str:
        try:
            api_url = f"{self.base_url}/video/generations"
            headers = self._build_headers()
            payload = self._build_request_payload(
                prompt=prompt,
                model=model,
                aspect_ratio=aspect_ratio,
                duration=duration,
                input_image_data=input_images,
                input_video_data=input_videos,
                **kwargs
            )

            # Debug: print payload content types (truncate base64 for readability)
            debug_content = []
            for item in payload.get("content", []):
                item_copy = dict(item)
                if item_copy.get("type") == "image_url":
                    url = item_copy.get("image_url", {}).get("url", "")
                    item_copy["image_url"] = {"url": url[:80] + "..." if len(url) > 80 else url}
                debug_content.append(item_copy)
            print(f"🎥 Starting CFGPU video generation, model: {model}, content: {debug_content}")

            async with HttpClient.create_aiohttp() as session:
                async with session.post(api_url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        try:
                            error_data = await response.json()
                        except Exception:
                            error_data = await response.text()
                        raise Exception(f"CFGPU video generation task creation failed: {error_data}")

                    result = await response.json()
                    task_id = result.get("id") or result.get("task_id")

                if not task_id:
                    raise Exception(f"CFGPU video generation task creation failed: {result}")

                print(f"🎥 CFGPU task created, task_id: {task_id}")

            video_url = await self._poll_task_status(task_id, model, headers)
            print(f"🎥 CFGPU video generation completed, video URL: {video_url}")
            return video_url

        except Exception as e:
            print(f"🎥 Error generating video with CFGPU: {str(e)}")
            traceback.print_exc()
            raise e
