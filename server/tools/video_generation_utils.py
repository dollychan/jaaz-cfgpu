# from engineio import payload
from utils.http_client import HttpClient

import aiofiles
import io
import os
import base64
from PIL import Image

from nanoid import generate
from mimetypes import guess_type
# import httpx
import mimetypes
from pymediainfo import MediaInfo
from PIL import Image


from services.config_service import FILES_DIR


def generate_video_file_id():
    return "vi_" + generate(size=8)


def generate_audio_file_id():
    return "au_" + generate(size=8)


async def get_video_info_and_save(
    url: str, file_path_without_extension: str
) -> tuple[str, int, int, str]:
    # Fetch the video asynchronously
    async with HttpClient.create_aiohttp() as session:
        async with session.get(url) as response:
            video_content = await response.read()

    # Save to temporary mp4 file first
    temp_path = f"{file_path_without_extension}.mp4"
    async with aiofiles.open(temp_path, "wb") as out_file:
        await out_file.write(video_content)
    print("🎥 Video saved to", temp_path)

    try:
        media_info = MediaInfo.parse(temp_path)
        for track in media_info.tracks:  # type: ignore
            if track.track_type == "Video":
                width = track.width
                height = track.height
                print(f"Width: {width}, Height: {height}")

        extension = "mp4"  # 默认使用 mp4，实际情况可以根据 codec_name 灵活判断

        # Get mime type
        mime_type = mimetypes.types_map.get(".mp4", "video/mp4")

        print(
            f"🎥 Video info - width: {width}, height: {height}, mime_type: {mime_type}, extension: {extension}"
        )

        return mime_type, width, height, extension
    except Exception as e:
        print(f"Error probing video file {temp_path}: {str(e)}")
        raise e


def get_video_base64(video_name: str) -> str:
    """Convert a local video file to base64 data URL"""
    video_path = os.path.join(FILES_DIR, video_name)
    with open(video_path, 'rb') as f:
        video_data = f.read()
    b64 = base64.b64encode(video_data).decode('utf-8')
    mime_type, _ = guess_type(video_path)
    if not mime_type:
        mime_type = 'video/mp4'
    return f"data:{mime_type};base64,{b64}"


def get_image_base64(image_name: str):
    # Process image
    image_path = os.path.join(FILES_DIR, f"{image_name}")
    image = Image.open(image_path)

    # 可爱的豆包，鲁棒性太拉了，拉的想骂人(图片支支持0.4-2.5比例的)
    # Kawaii Doubao video model has a fxxking bad robustness,
    # it can only handle images with aspect ratio between 0.4 and 2.5.

    width, height = image.size
    ratio = width / height
    if ratio > 2.5 or ratio < 0.4:
        # 宽高比大于2.5或者小于0.4的图片，现在只能暴力裁掉
        if ratio < 1:
            # 竖版图片
            new_height = int(width * 2.4)
            new_width = width
            image = image.resize(  # type:ignore
                (new_width, new_height), Image.Resampling.LANCZOS
            )
        elif ratio > 1:
            new_width = int(height * 2.4)
            new_height = height
            image = image.resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )
    else:
        new_width, new_height = image.size

    # 计算缩放因子，确保类型为float
    scale_factor: float = float(
        (float(1048576) / float(new_width * new_height)) ** 0.5
    )

    preview_image_width = int(new_width * scale_factor)
    preview_image_height = int(new_height * scale_factor)

    img = image.resize(
        (preview_image_width, preview_image_height), Image.Resampling.LANCZOS
    )
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")

    b64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
    mime_type, _ = guess_type(image_path)
    if not mime_type:
        mime_type = "image/png"
    return f"data:{mime_type};base64,{b64}"
