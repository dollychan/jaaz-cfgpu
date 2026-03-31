import os
import uuid
import mimetypes
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import aiofiles
import httpx
from services.config_service import MATERIALS_DIR, config_service
from services.db_service import db_service

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
    """Return (api_key, base_url, project_name) from the shared cfgpu config section."""
    cfg = config_service.app_config.get("cfgpu", {})
    api_key = cfg.get("api_key", "")
    text_url = cfg.get("url", "https://www.cfgpu.com/userapi/v1/model/v1/").rstrip("/")
    base_url = text_url.removesuffix("/model/v1")
    project_name = cfg.get("project_name", "default") or "default"
    return api_key, base_url, project_name


async def _get_asset(asset_id: str) -> dict:
    """Call CFGPU assets status API and return the result dict."""
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


async def _create_asset(public_url: str, asset_type: str) -> dict:
    """Call CFGPU assets create API.

    Returns the full Result dict from CFGPU, which may include:
    Id, GroupId, Status, CreateTime, AssetType, UpdateTime, ProjectName, URL
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
    return result.get("Result") or {}


@router.post("/upload")
async def upload_material(file: UploadFile = File(...)):
    _ensure_dir()
    content_type = file.content_type or ""
    if content_type not in ALLOWED_TYPES:
        ext = os.path.splitext(file.filename or "")[1].lower()
        content_type = _EXT_TO_MIME.get(ext, content_type)
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {content_type}")

    original_name = file.filename or "unnamed"
    ext = os.path.splitext(original_name)[1] or mimetypes.guess_extension(content_type) or ""

    # Save with temp UUID name first so CFGPU can fetch it before we know the asset ID
    tmp_name = f"{uuid.uuid4().hex}{ext}"
    tmp_dest = os.path.join(MATERIALS_DIR, tmp_name)

    async with aiofiles.open(tmp_dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            await f.write(chunk)

    ftype = _file_type(tmp_name)
    asset_type = _ASSET_TYPE_MAP.get(ftype, "Image")

    asset_id = None
    final_name = tmp_name
    final_dest = tmp_dest
    cfgpu_result: dict = {}

    api_key, _, _ = _get_cfgpu_config()
    public_base = os.environ.get("JAAZ_SERVER_URL", "").rstrip("/")

    if api_key and public_base:
        public_url = f"{public_base}/api/material/serve/{tmp_name}"
        try:
            cfgpu_result = await _create_asset(
                public_url=public_url,
                asset_type=asset_type,
            )
            asset_id = (
                cfgpu_result.get("Id")
                or cfgpu_result.get("id")
                or cfgpu_result.get("assetId")
            )
            if not asset_id:
                raise ValueError(f"CreateAsset returned no asset ID: {cfgpu_result}")

            # Rename local file to use asset ID as disk name (e.g. asset-xxx.jpg)
            final_name = f"{asset_id}{ext}"
            final_dest = os.path.join(MATERIALS_DIR, final_name)
            os.rename(tmp_dest, final_dest)

            # Insert DB record — original filename as display name
            await db_service.insert_asset_file(
                name=original_name,
                asset_id=asset_id,
                asset_type=cfgpu_result.get("AssetType", asset_type),
                group_id=cfgpu_result.get("GroupId", ""),
                project_name=cfgpu_result.get("ProjectName", "default"),
                status=cfgpu_result.get("Status", "Processing"),
            )
            print(f"✅ Asset created: {asset_id}, name: {original_name}")
        except Exception as e:
            print(f"⚠️ CreateAsset failed (file kept with tmp name): {e}")
    else:
        # No CFGPU configured — keep tmp name, no DB record
        print("ℹ️ No CFGPU api_key/public_base configured, file saved locally only")

    stat = os.stat(final_dest)
    return {
        "success": True,
        "name": final_name,
        "display_name": original_name,
        "asset_id": asset_id,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "type": ftype,
        "url": f"/api/material/serve/{final_name}",
    }


@router.get("/files")
async def list_materials():
    """Return all asset files from DB, enriched with local file info."""
    _ensure_dir()
    records = await db_service.get_all_asset_files()
    results = []
    for rec in records:
        asset_id = rec["asset_id"]
        # Find matching disk file: asset_id + any extension
        disk_name = None
        for fname in os.listdir(MATERIALS_DIR):
            if fname.startswith("."):
                continue
            stem = os.path.splitext(fname)[0]
            if stem == asset_id:
                disk_name = fname
                break

        # Always return complete record with all fields
        item = {
            "fid": rec.get("fid"),
            "name": rec.get("name", "Unknown"),
            "asset_id": rec.get("asset_id", ""),
            "group_id": rec.get("group_id", ""),
            "status": rec.get("status", "Unknown"),
            "asset_type": rec.get("asset_type", "Image"),
            "project_name": rec.get("project_name", "default"),
            "url": rec.get("url", ""),
            "created_at": rec.get("created_at", ""),
            "updated_at": rec.get("updated_at", ""),
            "disk_name": disk_name,
            "file_type": None,
        }

        if disk_name is None:
            # File missing on disk
            item["url"] = None
            item["size"] = None
            item["mtime"] = None
        else:
            full = os.path.join(MATERIALS_DIR, disk_name)
            try:
                stat = os.stat(full)
                item["url"] = f"/api/material/serve/{disk_name}"
                item["size"] = stat.st_size
                item["mtime"] = stat.st_mtime
                item["file_type"] = _file_type(disk_name)
            except OSError:
                item["url"] = None
                item["size"] = None
                item["mtime"] = None

        results.append(item)

    return results


@router.post("/poll-status")
async def poll_material_statuses():
    """Query CFGPU for every Processing asset and update DB.

    Returns:
        updated: list of asset_ids whose status changed in this call
        records: full list of all asset_file DB records (after update)
    """
    processing = await db_service.get_asset_files_by_status("Processing")

    api_key, _, _ = _get_cfgpu_config()
    updated: list[str] = []

    if api_key and processing:
        for rec in processing:
            asset_id = rec["asset_id"]
            try:
                result = await _get_asset(asset_id)
                # Response shape: { "Result": { "Status": "...", "URL": "...", ... } }
                inner = result.get("Result") or result
                status = inner.get("Status") or inner.get("status")
                url = inner.get("URL") or inner.get("url") or ""
                group_id = inner.get("GroupId") or inner.get("group_id") or ""
                if status:
                    await db_service.update_asset_file(
                        asset_id=asset_id,
                        status=status,
                        url=url,
                        group_id=group_id or None,
                    )
                    updated.append(asset_id)
                    print(f"📊 Asset {asset_id} → {status}")
            except Exception as e:
                print(f"⚠️ GetAsset failed for {asset_id}: {e}")

    all_records = await db_service.get_all_asset_files()
    return {"updated": updated, "records": all_records}


class RenameRequest(BaseModel):
    name: str


@router.patch("/rename/{asset_id}")
async def rename_material(asset_id: str, body: RenameRequest):
    """Rename the display name of an asset."""
    rec = await db_service.get_asset_file_by_asset_id(asset_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Asset not found")
    await db_service.rename_asset_file(asset_id, body.name.strip())
    return {"success": True}


@router.delete("/{asset_id}")
async def delete_material(asset_id: str):
    """Delete an asset record from DB (and optionally from disk)."""
    rec = await db_service.get_asset_file_by_asset_id(asset_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Asset not found")
    # Try to remove disk file
    _ensure_dir()
    for fname in os.listdir(MATERIALS_DIR):
        if fname.startswith("."):
            continue
        if os.path.splitext(fname)[0] == asset_id:
            try:
                os.remove(os.path.join(MATERIALS_DIR, fname))
            except OSError as e:
                print(f"⚠️ Could not remove disk file {fname}: {e}")
            break
    await db_service.delete_asset_file(asset_id)
    return {"success": True}


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
