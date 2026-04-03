import os
import traceback
from PIL import Image, PngImagePlugin
from io import BytesIO
import base64
import json
from typing import Any, Optional, Tuple
from nanoid import generate
from utils.http_client import HttpClient
from services.config_service import FILES_DIR


def generate_image_id() -> str:
    """Generate unique image ID"""
    return generate(size=10)


async def get_image_info_and_save(
    url: str,
    file_path_without_extension: str,
    is_b64: bool = False,
    metadata: Optional[dict[str, Any]] = None
) -> Tuple[str, int, int, str]:
    """
    Download image from URL or decode base64, convert to PNG and save with metadata

    Args:
        url: Image URL or base64 string
        file_path_without_extension: File path without extension
        is_b64: Whether the url is a base64 string
        metadata: Optional metadata to be saved in PNG info

    Returns:
        tuple[str, int, int, str]: (mime_type, width, height, extension) - always PNG
    """
    try:
        if is_b64:
            image_data = base64.b64decode(url)
        else:
            # Fetch the image asynchronously
            async with HttpClient.create_aiohttp() as session:
                async with session.get(url) as response:
                    # Read the image content as bytes
                    image_data = await response.read()

        # Open image to get info
        image = Image.open(BytesIO(image_data))
        width, height = image.size
        
        # Store original format for debugging
        original_format = image.format or 'Unknown'
        print(f"Converting {original_format} image to PNG: {width}x{height}")

        # Handle different color modes properly for PNG conversion
        if image.mode == 'P':
            # Palette mode - convert to RGBA to preserve potential transparency
            if 'transparency' in image.info:
                image = image.convert('RGBA')
            else:
                image = image.convert('RGB')
        elif image.mode == 'LA':
            # Grayscale with alpha - convert to RGBA
            image = image.convert('RGBA')
        elif image.mode == 'L':
            # Grayscale - can stay as L or convert to RGB
            # PNG supports grayscale, so we can keep it
            pass
        elif image.mode == 'CMYK':
            # CMYK mode - convert to RGB
            image = image.convert('RGB')
        elif image.mode in ('RGB', 'RGBA'):
            # Already compatible with PNG
            pass
        else:
            # For any other modes, convert to RGB as a safe fallback
            print(f"Warning: Unusual color mode {image.mode}, converting to RGB")
            image = image.convert('RGB')

        # Unified format: always PNG
        extension = 'png'
        mime_type = 'image/png'

        # Prepare PNG info for metadata
        pnginfo = PngImagePlugin.PngInfo()
        
        # Add original format info
        pnginfo.add_text("original_format", original_format)
        
        if metadata:
            for key, value in metadata.items():
                try:
                    # Handle different value types
                    if isinstance(value, (dict, list)):
                        # Serialize complex types as JSON
                        text_value = json.dumps(value, ensure_ascii=False)
                    elif value is None:
                        text_value = "null"
                    else:
                        # Convert to string
                        text_value = str(value)
                    
                    pnginfo.add_text(str(key), text_value)
                except Exception as e:
                    print(f"Warning: Failed to add metadata key '{key}': {e}")
                    traceback.print_stack()

        # Save as PNG with metadata
        file_path = f"{file_path_without_extension}.{extension}"
        
        # Save with optimizations and metadata
        if metadata or original_format != 'PNG':
            image.save(file_path, format='PNG', optimize=True, pnginfo=pnginfo)
        else:
            image.save(file_path, format='PNG', optimize=True)
        
        print(f"Successfully saved as PNG: {file_path}")
        return mime_type, width, height, extension

    except Exception as e:
        print(f"Error processing image: {e}")
        raise e


# Canvas-related utilities have been moved to tools/image_generation/image_canvas_utils.py


# Canvas element generation moved to tools/image_generation/image_canvas_utils.py


# Canvas saving functionality moved to tools/image_generation/image_canvas_utils.py


# Image generation orchestration moved to tools/image_generation/image_generation_core.py
# Notification functions moved to tools/image_generation/image_canvas_utils.py


def process_input_image(input_image: str | None) -> str | None:
    """
    Resolve an input image reference to a URL accessible by external APIs.

    Mirrors the same strategy as video_generation_core._resolve_local_to_server_url:
    - http/https/data: URLs → returned as-is
    - /api/file/... or /api/material/serve/... (relative API paths) →
        prefixed with JAAZ_SERVER_URL
    - asset-xxx / asset://xxx (material library assets) →
        returned as asset://xxx for CFGPU native asset references
    - bare local filename → resolved via FILES_DIR scan →
        JAAZ_SERVER_URL/api/file/<fname>
    """
    if not input_image:
        return None

    # Already absolute HTTP URL or data URI
    if input_image.startswith(('http://', 'https://', 'data:')):
        return input_image

    # Material library asset → asset:// protocol (CFGPU native reference)
    if input_image.startswith('asset-') or input_image.startswith('asset://'):
        asset_id = input_image.removeprefix('asset://')
        url = f"asset://{asset_id}"
        print(f"🔗 Image input material asset '{input_image}' → {url}")
        return url

    server_base = os.environ.get(
        "JAAZ_SERVER_URL",
        f"http://127.0.0.1:{os.environ.get('DEFAULT_PORT', '57988')}"
    ).rstrip("/")

    # Relative API path (e.g. /api/file/xxx or /api/material/serve/xxx)
    if input_image.startswith('/api/'):
        url = f"{server_base}{input_image}"
        print(f"🔗 Image input API path '{input_image}' → {url}")
        return url

    # Bare local filename — scan FILES_DIR for a match
    ref_stem = os.path.splitext(input_image)[0]
    try:
        for fname in os.listdir(FILES_DIR):
            if fname == input_image or os.path.splitext(fname)[0] == ref_stem:
                url = f"{server_base}/api/file/{fname}"
                print(f"🔗 Image input local file '{input_image}' → {url}")
                return url
    except OSError:
        pass

    print(f"⚠️ Image input '{input_image}' not found in FILES_DIR, passing as-is")
    return input_image
