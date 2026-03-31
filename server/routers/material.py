import json
import os
import uuid
import mimetypes
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import aiofiles
import httpx
from services.config_service import MATERIALS_DIR, config_service

router = APIRouter(prefix="/api/material")

ALLOWED_TYPES = {
    # Images
    "image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp",
    "image/tiff", "image/heic", "image/heif",
    # Videos
    "video/mp4", "video/webm", "video/quicktime", "video/x-msvideo",
    "video/x-matroska", "video/mpeg",
    # Audio — include all common browser/OS variants for MP3 and friends
    "audio/mpeg", "audio/mp3", "audio/x-mpeg", "audio/mpeg3", "audio/x-mpeg-3",
    "audio/wav", "audio/x-wav", "audio/wave",
    "audio/aac", "audio/x-aac",
    "audio/flac", "audio/x-flac",
    "audio/m4a", "audio/x-m4a", "audio/mp4",
    "audio/ogg", "audio/opus",
}

# Extension → canonical MIME type, used when browser sends
# application/octet-stream or an unrecognised MIME type.
_EXT_TO_MIME: dict[str, str] = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".gif": "image/gif",
    ".webp": "image/webp", ".bmp": "image/bmp",
    ".tiff": "image/tiff", ".tif": "image/tiff",
    ".heic": "image/heic", ".heif": "image/heif",
    ".mp4": "video/mp4", ".webm": "video/webm",
    ".mov": "video/quicktime", ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".m4a": "audio/x-m4a",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
}

_ASSET_TYPE_MAP = {
    "image": "Image",
    "video": "Video",
    "audio": "Audio",
}


def _ensure_dir():
    os.makedirs(MATERIALS_DIR, exist_ok=True)


def _file_type(name: str) -> str:
    ext = os.path.splitext(name)[1].lower()
    if ext in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp",
               ".tiff", ".tif", ".heic", ".heif"}:
        return "image"
    if ext in {".mp4", ".webm", ".mov", ".avi", ".mkv", ".mpeg", ".mpg"}:
        return "video"
    if ext in {".mp3", ".wav", ".aac", ".flac", ".m4a", ".ogg", ".opus"}:
        return "audio"
    return "file"


def _get_cfgpu_config() -> tuple[str, str, str]:
    """Return (api_key, base_url) from the shared cfgpu config section.

    base_url is derived the same way as CfgpuImageProvider / CfgpuVideoProvider
    so the user only needs to configure cfgpu once.
    """
    cfg = config_service.app_config.get("cfgpu", {})
    api_key = cfg.get("api_key", "")
    text_url = cfg.get("url", "https://www.cfgpu.com/userapi/v1/model/v1/").rstrip("/")
    base_url = text_url.removesuffix("/model/v1")
    project_name = cfg.get("project_name", "default") or "default"
    return api_key, base_url, project_name


# ---------- Status cache helpers ----------

def _status_cache_path() -> str:
    return os.path.join(MATERIALS_DIR, ".material_status.json")


def _load_status_cache() -> dict:
    path = _status_cache_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_status_cache(cache: dict) -> None:
    _ensure_dir()
    path = _status_cache_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


async def _get_asset(asset_id: str) -> dict:
    """Call CFGPU assets status API and return the result dict.

    GET /userapi/v1/assets/status?assetId=<assetId>
    """
    api_key, base_url, _ = _get_cfgpu_config()
    if not api_key:
        raise ValueError("cfgpu api_key is not configured")

    url = f"{base_url}/assets/status?assetId={asset_id}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()


async def _create_asset(public_url: str, asset_type: str) -> str:
    """Call CFGPU assets create API and return the asset ID.

    POST /userapi/v1/assets/create  { url, assetType, projectName }
    """
    api_key, base_url, project_name = _get_cfgpu_config()
    if not api_key:
        raise ValueError("cfgpu api_key is not configured")

    url = f"{base_url}/assets/create"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "url": public_url,
        "assetType": asset_type,
        "projectName": project_name,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

    # Response shape: { "Result": { "Id": "asset-..." }, "ResponseMetadata": {...} }
    asset_id = (
        (result.get("Result") or {}).get("Id")
        or result.get("assetId")
        or result.get("id")
        or result.get("Id")
    )
    if not asset_id:
        raise ValueError(f"CreateAsset returned no assetId: {result}")
    return asset_id


@router.post("/upload")
async def upload_material(file: UploadFile = File(...)):
    _ensure_dir()
    content_type = file.content_type or ""
    # Some browsers / Electron environments send application/octet-stream or an
    # unrecognised MIME variant — fall back to extension-based detection.
    if content_type not in ALLOWED_TYPES:
        ext = os.path.splitext(file.filename or "")[1].lower()
        content_type = _EXT_TO_MIME.get(ext, content_type)
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {content_type}")

    ext = os.path.splitext(file.filename or "")[1] or mimetypes.guess_extension(content_type) or ""
    # Save with temp UUID name first
    tmp_name = f"{uuid.uuid4().hex}{ext}"
    dest = os.path.join(MATERIALS_DIR, tmp_name)

    async with aiofiles.open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            await f.write(chunk)

    ftype = _file_type(tmp_name)
    stat = os.stat(dest)

    asset_id = None
    final_name = tmp_name
    final_dest = dest

    # Upload to CFGPU asset library when api_key is configured.
    # Use JAAZ_SERVER_URL env var so CFGPU can fetch the file from our server.
    api_key, _, _ = _get_cfgpu_config()
    public_base = os.environ.get("JAAZ_SERVER_URL", "").rstrip("/")

    if api_key and public_base:
        public_url = f"{public_base}/api/material/serve/{tmp_name}"
        try:
            asset_id = await _create_asset(
                public_url=public_url,
                asset_type=_ASSET_TYPE_MAP.get(ftype, "Image"),
            )
            # Rename local file to use asset ID
            final_name = f"{asset_id}{ext}"
            final_dest = os.path.join(MATERIALS_DIR, final_name)
            os.rename(dest, final_dest)
            stat = os.stat(final_dest)
            
            # Initialize status as Processing
            cache = _load_status_cache()
            cache[asset_id] = "Processing"
            _save_status_cache(cache)
            
            print(f"✅ Asset created: {asset_id}, status initialized as Processing")
        except Exception as e:
            print(f"⚠️ CreateAsset failed (file kept with tmp name): {e}")

    return {
        "success": True,
        "name": final_name,
        "original_name": file.filename,
        "asset_id": asset_id,
        "path": final_dest,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "type": ftype,
        "url": f"/api/material/serve/{final_name}",
    }


@router.get("/files")
async def list_materials():
    _ensure_dir()
    cache = _load_status_cache()
    results = []
    for name in sorted(os.listdir(MATERIALS_DIR)):
        if name.startswith("."):   # skip hidden files (.material_status.json etc.)
            continue
        full = os.path.join(MATERIALS_DIR, name)
        if not os.path.isfile(full):
            continue
        stat = os.stat(full)
        ftype = _file_type(name)
        stem = os.path.splitext(name)[0]
        asset_id = stem if stem.startswith("asset-") else None
        status = cache.get(asset_id) if asset_id else None
        results.append({
            "name": name,
            "asset_id": asset_id,
            "display_name": asset_id or name,
            "path": full,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "type": ftype,
            "url": f"/api/material/serve/{name}",
            "status": status,
        })
    results.sort(key=lambda x: x["mtime"], reverse=True)
    return results


@router.post("/poll-status")
async def poll_material_statuses():
    """Query assets status API for every Processing (or unknown) asset and update the local cache.

    Returns:
        statuses: full {asset_id: status} cache
        updated:  only the asset_ids whose status changed in this call
    """
    _ensure_dir()
    cache = _load_status_cache()

    # Collect all asset IDs present as local files
    asset_ids = []
    for name in os.listdir(MATERIALS_DIR):
        if name.startswith("."):
            continue
        stem = os.path.splitext(name)[0]
        if stem.startswith("asset-"):
            asset_ids.append(stem)

    # Only poll assets whose status is still mutable (Processing or unknown)
    to_poll = [aid for aid in asset_ids if cache.get(aid) not in ("Active", "Failed")]

    updated: dict = {}
    api_key, _, _ = _get_cfgpu_config()

    if api_key and to_poll:
        for asset_id in to_poll:
            try:
                result = await _get_asset(asset_id)
                # Response shape mirrors CreateAsset: { "Result": { "Status": "..." } }
                status = (
                    (result.get("Result") or {}).get("Status")
                    or (result.get("Result") or {}).get("status")
                    or result.get("status")
                    or result.get("Status")
                )
                if status:
                    cache[asset_id] = status
                    updated[asset_id] = status
                    print(f"📊 Asset {asset_id} status: {status}")
            except Exception as e:
                print(f"⚠️ GetAsset failed for {asset_id}: {e}")

        if updated:
            _save_status_cache(cache)

    return {"statuses": cache, "updated": updated}


@router.get("/serve/{filename}")
async def serve_material(filename: str):
    _ensure_dir()
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = os.path.join(MATERIALS_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    mime, _ = mimetypes.guess_type(path)
    return FileResponse(path, media_type=mime or "application/octet-stream")
