import json
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
    """Call volcengine GetAsset API and return the result dict."""
    cfg = _get_ml_config()
    ak = cfg.get("ak", "")
    sk = cfg.get("sk", "")
    project_name = cfg.get("project_name", "default") or "default"

    if not ak or not sk:
        raise ValueError("material_library AK/SK are not configured")

    payload: dict = {"Id": asset_id}
    if project_name and project_name != "default":
        payload["ProjectName"] = project_name

    result = await volcengine_post(
        ak=ak, sk=sk,
        service="ark",
        action="GetAsset",
        version="2024-01-01",
        body=payload,
    )
    return result


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
    """Query GetAsset for every Processing (or unknown) asset and update the local cache.

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
    cfg = _get_ml_config()
    if cfg.get("ak") and cfg.get("sk") and to_poll:
        for asset_id in to_poll:
            try:
                result = await _get_asset(asset_id)
                data = result.get("Result") or result
                status = data.get("Status")
                if status:
                    cache[asset_id] = status
                    updated[asset_id] = status
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
