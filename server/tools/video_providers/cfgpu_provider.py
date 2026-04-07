import asyncio
import traceback
from typing import Optional, Dict, Any, List

from .video_base_provider import VideoProviderBase
from utils.http_client import HttpClient
from services.config_service import config_service


class ContentPolicyError(Exception):
    """Raised when CFGPU rejects a request due to content policy. Should not be retried."""
    pass


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

    # doubao-seedance-2-0 in r2v mode: output pixel count must be ≤ this value
    _R2V_MAX_PIXELS = 927408  # 1280×725 ≈ limit; 1280×720 = 921600 is safe

    # Resolution string → (short_side, max_safe) mapping
    # When input images are present (r2v), the API derives output size from them;
    # passing an explicit resolution caps the output.
    _RESOLUTION_MAP: Dict[str, str] = {
        "480p": "480p",
        "720p": "720p",
        "1080p": "720p",  # 1080p → 1920×1080 ≈ 2.07M pixels > r2v limit; cap to 720p
    }

    def _build_request_payload(
        self,
        prompt: str,
        model: str,
        aspect_ratio: str = "16:9",
        duration: int = 5,
        resolution: str = "480p",
        input_image_data: Optional[List[str]] = None,
        input_video_data: Optional[List[str]] = None,
        input_audio_data: Optional[List[str]] = None,
        image_role: str = "auto",
        generate_audio: bool = True,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        image_role controls how input images are submitted to the API.
        The API defines three mutually exclusive image scenarios:
          "auto"            – infer from count: 1→first_frame, 2→first_last_frame, 3+→reference_image
          "first_frame"     – 1 image used as the starting frame of the video
          "first_last_frame"– 2 images used as starting and ending frames (bookend)
          "reference_image" – 1–9 images used as style/content references (multimodal)
        """
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        # API limits (multimodal mode): reference_image 0–9, reference_video 0–3, reference_audio 0–3
        if input_image_data and len(input_image_data) > 9:
            print(f"⚠️ CFGPU: trimming input_images from {len(input_image_data)} to 9 (API max)")
            input_image_data = input_image_data[:9]
        if input_video_data and len(input_video_data) > 3:
            print(f"⚠️ CFGPU: trimming input_videos from {len(input_video_data)} to 3 (API max)")
            input_video_data = input_video_data[:3]
        if input_audio_data and len(input_audio_data) > 3:
            print(f"⚠️ CFGPU: trimming input_audios from {len(input_audio_data)} to 3 (API max)")
            input_audio_data = input_audio_data[:3]

        if input_image_data:
            n = len(input_image_data)
            # first_frame / first_last_frame are mutually exclusive with reference_video
            # and reference_audio — they cannot appear in the same request.
            # When other media is present we must use reference_image (multimodal mode).
            has_other_media = bool(input_video_data or input_audio_data)

            if image_role == "auto":
                if has_other_media or n >= 3:
                    effective_role = "reference_image"
                elif n == 1:
                    effective_role = "first_frame"
                else:  # n == 2
                    effective_role = "first_last_frame"
            elif image_role in ("first_frame", "first_last_frame") and has_other_media:
                # Caller explicitly chose a frame role but also supplied other media —
                # override to reference_image to avoid an API InvalidParameter error.
                print(
                    f"⚠️ image_role='{image_role}' is incompatible with reference_video/"
                    "reference_audio; overriding to 'reference_image'"
                )
                effective_role = "reference_image"
            else:
                effective_role = image_role

            if effective_role == "first_last_frame":
                content.append({"type": "image_url", "image_url": {"url": input_image_data[0]}, "role": "first_frame"})
                if n >= 2:
                    content.append({"type": "image_url", "image_url": {"url": input_image_data[1]}, "role": "last_frame"})
                # Additional images beyond 2 are treated as reference_image
                for img_url in input_image_data[2:]:
                    content.append({"type": "image_url", "image_url": {"url": img_url}, "role": "reference_image"})
            else:
                for img_url in input_image_data:
                    content.append({"type": "image_url", "image_url": {"url": img_url}, "role": effective_role})

        if input_video_data:
            for vid_url in input_video_data:
                content.append({"type": "video_url", "video_url": {"url": vid_url}, "role": "reference_video"})

        if input_audio_data:
            for aud_url in input_audio_data:
                content.append({"type": "audio_url", "audio_url": {"url": aud_url}, "role": "reference_audio"})

        # 当有音频引用时，强制打开音频合成，避免上游默认 false 导致未混音的问题
        if input_audio_data and not generate_audio:
            print("⚠️ Detected input_audio_data but generate_audio=False; overriding to True")
            generate_audio = True

        # In r2v mode (any input images present) the API derives the output video
        # resolution from the input image dimensions.  doubao-seedance-2-0 hard-limits
        # the output to ≤ 927408 pixels (≈ 1280×720).  Pass an explicit resolution so
        # the API caps the output instead of deriving it from a potentially larger image.
        has_r2v = bool(input_image_data)
        effective_resolution = self._RESOLUTION_MAP.get(resolution, resolution)
        if has_r2v and resolution == "1080p":
            print(
                f"⚠️ CFGPU r2v mode: capping resolution 1080p → 720p "
                f"(API limit: ≤{self._R2V_MAX_PIXELS} pixels)"
            )

        print(f"🎵 CFGPU payload: model={model}, resolution={effective_resolution}, generate_audio={generate_audio}, r2v={has_r2v}, videos={len(input_video_data or [])}, audios={len(input_audio_data or [])}")

        return {
            "model": model,
            "content": content,
            "generate_audio": generate_audio,
            "ratio": aspect_ratio,
            "resolution": effective_resolution,
            "duration": duration,
            "watermark": False,
        }

    async def _poll_task_status(
        self,
        task_id: str,
        model: str,
        headers: Dict[str, str],
        timeout_seconds: int = 7200,  # 2 hours
        poll_interval: int = 5,
    ) -> str:
        """Poll task status until completion or timeout."""
        polling_url = f"{self.base_url}/video/tasks/{task_id}"
        status = "pending"
        max_polls = timeout_seconds // poll_interval
        poll_count = 0

        async with HttpClient.create_aiohttp() as session:
            while status not in ("succeeded", "failed", "cancelled"):
                if poll_count >= max_polls:
                    raise Exception(
                        f"CFGPU task {task_id} timed out after {timeout_seconds}s"
                    )
                print(f"🎥 Polling CFGPU generation {task_id} ({poll_count}/{max_polls}), status: {status} ...")
                await asyncio.sleep(poll_interval)
                poll_count += 1

                async with session.get(polling_url, headers=headers) as poll_response:
                    raw_text = await poll_response.text()
                    # print(f"🔍 CFGPU poll #{poll_count} HTTP {poll_response.status}, body: {raw_text[:500]}")
                    if poll_response.status != 200:
                        raise Exception(f"CFGPU poll failed with HTTP {poll_response.status}: {raw_text}")
                    try:
                        import json as _json
                        poll_res = _json.loads(raw_text) if raw_text.strip() else {}
                    except Exception:
                        raise Exception(f"CFGPU poll returned non-JSON (HTTP {poll_response.status}): {raw_text[:300]}")
                    if poll_res is None:
                        poll_res = {}
                    status = poll_res.get("status", "pending")

                    if status == "succeeded":
                        # 🔍 LOG: full succeeded response to confirm field names
                        print(f"🔍 CFGPU succeeded response (full): {poll_res}")
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
                        error_obj = poll_res.get("error") or {}
                        error_code = error_obj.get("code") if isinstance(error_obj, dict) else None
                        detail = (
                            error_obj if isinstance(error_obj, dict) and error_obj
                            else poll_res.get("message")
                            or poll_res.get("error_message")
                            or poll_res.get("reason")
                            or (poll_res.get("content") or {}).get("message")
                            or f"Task {status} with no details"
                        )
                        print(f"🎥 CFGPU task failed, full response: {poll_res}")
                        # Content policy errors are non-retriable — raise a distinct type
                        # so the caller can return a user-facing message instead of retrying.
                        NON_RETRIABLE_PREFIXES = (
                            "OutputVideoSensitiveContentDetected",
                            "InputSensitiveContentDetected",
                            "InputImageSensitiveContentDetected",
                            "ContentPolicyViolation",
                        )
                        if error_code and any(error_code.startswith(p) for p in NON_RETRIABLE_PREFIXES):
                            raise ContentPolicyError(
                                f"Content policy violation ({error_code}): the prompt or input media was rejected. "
                                f"STOP. Do NOT retry with any other tool. "
                                f"Tell the user their content was rejected and ask them to change the prompt or input media."
                            )
                        raise Exception(f"CFGPU video generation failed: {detail}")

        raise Exception(f"Task polling finished with unexpected status: {status}")

    async def generate(
        self,
        prompt: str,
        model: str,
        resolution: str = "480p",
        duration: int = 5,
        aspect_ratio: str = "16:9",
        input_images: Optional[List[str]] = None,
        input_videos: Optional[List[str]] = None,
        input_audios: Optional[List[str]] = None,
        camera_fixed: bool = True,
        image_role: str = "auto",
        generate_audio: bool = True,
        **kwargs: Any
    ) -> str:
        # cfgpu API constraint: reference_audio cannot be the only reference input
        if input_audios and not input_images and not input_videos:
            raise ValueError(
                "Audio reference requires at least one image or video reference. "
                "Please provide input_images or input_videos alongside input_audios."
            )
        max_retries = 2
        last_exc: Exception = RuntimeError("unreachable")
        for attempt in range(1, max_retries + 1):
            try:
                api_url = f"{self.base_url}/video/generations"
                headers = self._build_headers()
                payload = self._build_request_payload(
                    prompt=prompt,
                    model=model,
                    aspect_ratio=aspect_ratio,
                    duration=duration,
                    resolution=resolution,
                    input_image_data=input_images,
                    input_video_data=input_videos,
                    input_audio_data=input_audios,
                    image_role=image_role,
                    generate_audio=generate_audio,
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
                payload_meta = {k: v for k, v in payload.items() if k != "content"}
                print(f"🎥 Starting CFGPU video generation (attempt {attempt}/{max_retries}), model: {model}, payload_fields: {payload_meta}, content: {debug_content}")
                # 🔍 LOG: confirm exact field names sent to API
                print(f"🔍 CFGPU request payload keys & values (non-content): ratio={payload.get('ratio')!r}, duration={payload.get('duration')!r}, generate_audio={payload.get('generate_audio')!r}, watermark={payload.get('watermark')!r}")

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
                        # Log full creation response to inspect which fields are accepted/echoed
                        print(f"🎥 CFGPU task creation response (full): {result}")

                    if not task_id:
                        raise Exception(f"CFGPU video generation task creation failed: {result}")

                    print(f"🎥 CFGPU task created, task_id: {task_id}")

                video_url = await self._poll_task_status(task_id, model, headers)
                print(f"🎥 CFGPU video generation completed, video URL: {video_url}")
                return video_url

            except ContentPolicyError:
                # Non-retriable: surface directly to the tool caller
                raise
            except Exception as e:
                last_exc = e
                err_str = str(e)
                is_internal = "InternalServiceError" in err_str
                if is_internal and attempt < max_retries:
                    wait = 10 * attempt
                    print(f"🎥 CFGPU InternalServiceError on attempt {attempt}, retrying in {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                print(f"🎥 Error generating video with CFGPU: {err_str}")
                traceback.print_exc()
                raise e
        raise last_exc
