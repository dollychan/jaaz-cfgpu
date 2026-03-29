from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from common import DEFAULT_PORT
from tools.utils.image_canvas_utils import generate_file_id
from tools.video_generation_utils import generate_video_file_id
from services.config_service import FILES_DIR

from PIL import Image
from io import BytesIO
import os
from fastapi import APIRouter, HTTPException, UploadFile, File
import httpx
import aiofiles
from mimetypes import guess_type
from utils.http_client import HttpClient

router = APIRouter(prefix="/api")
os.makedirs(FILES_DIR, exist_ok=True)


@router.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    print('🎥 upload_video file', file.filename)
    file_id = generate_video_file_id()
    filename = file.filename or 'video.mp4'

    # Determine extension from mime type or filename
    content_type = file.content_type or ''
    if 'mp4' in content_type or filename.endswith('.mp4'):
        extension = 'mp4'
    elif 'mov' in content_type or filename.endswith('.mov'):
        extension = 'mov'
    elif 'webm' in content_type or filename.endswith('.webm'):
        extension = 'webm'
    else:
        extension = 'mp4'

    file_path = os.path.join(FILES_DIR, f'{file_id}.{extension}')
    content = await file.read()
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(content)

    print('🎥 upload_video saved to', file_path)
    return {
        'file_id': f'{file_id}.{extension}',
        'url': f'/api/file/{file_id}.{extension}',
    }

# 上传图片接口，支持表单提交
@router.post("/upload_image")
async def upload_image(file: UploadFile = File(...), max_size_mb: float = 3.0):
    print('🦄upload_image file', file.filename)
    # 生成文件 ID 和文件名
    file_id = generate_file_id()
    filename = file.filename or ''

    # Read the file content
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")
    original_size_mb = len(content) / (1024 * 1024)  # Convert to MB

    # Open the image from bytes to get its dimensions
    with Image.open(BytesIO(content)) as img:
        width, height = img.size
        
        # Check if compression is needed
        if original_size_mb > max_size_mb:
            print(f'🦄 Image size ({original_size_mb:.2f}MB) exceeds limit ({max_size_mb}MB), compressing...')
            
            # Convert to RGB if necessary (for JPEG compression)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create a white background for transparent images
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Compress the image
            compressed_content = compress_image(img, max_size_mb)
            
            # Save compressed image using Image.save
            extension = 'jpg'  # Force JPEG for compressed images
            file_path = os.path.join(FILES_DIR, f'{file_id}.{extension}')
            
            # Create new image from compressed content and save
            with Image.open(BytesIO(compressed_content)) as compressed_img:
                width, height = compressed_img.size
                await run_in_threadpool(compressed_img.save, file_path, format='JPEG', quality=95, optimize=True)
                # compressed_img.save(file_path, format='JPEG', quality=95, optimize=True)
            
            final_size_mb = len(compressed_content) / (1024 * 1024)
            print(f'🦄 Compressed from {original_size_mb:.2f}MB to {final_size_mb:.2f}MB')
        else:
            # Determine the file extension from original file
            mime_type, _ = guess_type(filename)
            if mime_type and mime_type.startswith('image/'):
                extension = mime_type.split('/')[-1]
                # Handle common image format mappings
                if extension == 'jpeg':
                    extension = 'jpg'
            else:
                extension = 'jpg'  # Default to jpg for unknown types
            
            # Save original image using Image.save
            file_path = os.path.join(FILES_DIR, f'{file_id}.{extension}')
            
            # Determine save format based on extension
            save_format = 'JPEG' if extension.lower() in ['jpg', 'jpeg'] else extension.upper()
            if save_format == 'JPEG':
                img = img.convert('RGB')
            
            # img.save(file_path, format=save_format)
            await run_in_threadpool(img.save, file_path, format=save_format)

    # 返回文件信息
    print('🦄upload_image file_path', file_path)
    return {
        'file_id': f'{file_id}.{extension}',
        'url': f'http://localhost:{DEFAULT_PORT}/api/file/{file_id}.{extension}',
        'width': width,
        'height': height,
    }


def compress_image(img: Image.Image, max_size_mb: float) -> bytes:
    """
    Compress an image to be under the specified size limit.
    """
    # Start with high quality
    quality = 95
    
    while quality > 10:
        # Save to bytes buffer
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        
        # Check size
        size_mb = len(buffer.getvalue()) / (1024 * 1024)
        
        if size_mb <= max_size_mb:
            return buffer.getvalue()
        
        # Reduce quality for next iteration
        quality -= 10
    
    # If still too large, try reducing dimensions
    original_width, original_height = img.size
    scale_factor = 0.8
    
    while scale_factor > 0.3:
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Try with moderate quality
        buffer = BytesIO()
        resized_img.save(buffer, format='JPEG', quality=70, optimize=True)
        
        size_mb = len(buffer.getvalue()) / (1024 * 1024)
        
        if size_mb <= max_size_mb:
            return buffer.getvalue()
        
        scale_factor -= 0.1
    
    # Last resort: very low quality
    buffer = BytesIO()
    resized_img.save(buffer, format='JPEG', quality=30, optimize=True)
    return buffer.getvalue()


# 文件下载接口
@router.get("/file/{file_id}")
async def get_file(file_id: str):
    file_path = os.path.join(FILES_DIR, f'{file_id}')
    print('🦄get_file file_path', file_path)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


@router.post("/comfyui/object_info")
async def get_object_info(data: dict):
    url = data.get('url', '')
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    try:
        timeout = httpx.Timeout(10.0)
        async with HttpClient.create(timeout=timeout) as client:
            response = await client.get(f"{url}/api/object_info")
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code, detail=f"ComfyUI server returned status {response.status_code}")
    except Exception as e:
        if "ConnectError" in str(type(e)) or "timeout" in str(e).lower():
            print(f"ComfyUI connection error: {str(e)}")
            raise HTTPException(
                status_code=503, detail="ComfyUI server is not available. Please make sure ComfyUI is running.")
        print(f"Unexpected error connecting to ComfyUI: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to connect to ComfyUI: {str(e)}")
