import os
import uuid
import mimetypes
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import aiofiles
from services.config_service import MATERIALS_DIR, config_service
from utils.volcengine_sign import volcengine_post

router = APIRouter(prefix="/api/material")

ALLOWED_TYPES = {
    "image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp",
    "video/mp4", "video/webm", "video/quicktime", "video/x-msvideo",
    "video/x-matroska", "video/mpeg",
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
    if ext in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}:
        return "image"
    if ext in {".mp4", ".webm", ".mov", ".avi", ".mkv", ".mpeg", ".mpg"}:
        return "video"
    if ext in {".mp3", ".wav", ".aac", ".flac", ".m4a", ".ogg"}:
        return "audio"
    return "file"


def _get_ml_config():
    return config_service.app_config.get("material_library", {})


async def _create_asset(public_url: str, asset_type: str, name: str) -> str:
    """Call volcengine CreateAsset API and return the asset ID."""
    cfg = _get_ml_config()
    ak = cfg.get("ak", "")
    sk = cfg.get("sk", "")
    group_id = cfg.get("group_id", "")
    project_name = cfg.get("project_name", "default") or "default"

    if not ak or not sk or not group_id:
        raise ValueError("material_library AK/SK/group_id are not configured")

    payload = {
        "GroupId": group_id,
        "URL": public_url,
        "AssetType": asset_type,
        "Name": name,
    }
    if project_name and project_name != "default":
        payload["ProjectName"] = project_name

    result = await volcengine_post(
        ak=ak, sk=sk,
        service="ark",
        action="CreateAsset",
        version="2024-01-01",
        body=payload,
    )
    asset_id = (result.get("Result") or result).get("Id") or result.get("Id")
    if not asset_id:
        raise ValueError(f"CreateAsset returned no Id: {result}")
    return asset_id


@router.post("/upload")
async def upload_material(file: UploadFile = File(...)):
    _ensure_dir()
    content_type = file.content_type or ""
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

    # TODO: 配置好 AK/SK/GroupId 后取消注释，将素材上传到火山引擎素材库
    # cfg = _get_ml_config()
    # public_base = (cfg.get("public_base_url") or "").rstrip("/")
    # if public_base and cfg.get("ak") and cfg.get("sk") and cfg.get("group_id"):
    #     public_url = f"{public_base}/api/material/serve/{tmp_name}"
    #     try:
    #         asset_id = await _create_asset(
    #             public_url=public_url,
    #             asset_type=_ASSET_TYPE_MAP.get(ftype, "Image"),
    #             name=file.filename or tmp_name,
    #         )
    #         # Rename local file to use asset ID
    #         final_name = f"{asset_id}{ext}"
    #         final_dest = os.path.join(MATERIALS_DIR, final_name)
    #         os.rename(dest, final_dest)
    #         stat = os.stat(final_dest)
    #     except Exception as e:
    #         print(f"⚠️ CreateAsset failed (file kept with tmp name): {e}")

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
    results = []
    for name in sorted(os.listdir(MATERIALS_DIR)):
        full = os.path.join(MATERIALS_DIR, name)
        if not os.path.isfile(full):
            continue
        stat = os.stat(full)
        ftype = _file_type(name)
        # Asset ID is the stem if it looks like an asset ID (starts with "asset-")
        stem = os.path.splitext(name)[0]
        asset_id = stem if stem.startswith("asset-") else None
        results.append({
            "name": name,
            "asset_id": asset_id,
            "display_name": asset_id or name,
            "path": full,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "type": ftype,
            "url": f"/api/material/serve/{name}",
        })
    results.sort(key=lambda x: x["mtime"], reverse=True)
    return results


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
