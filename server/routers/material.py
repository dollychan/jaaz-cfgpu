import os
import uuid
import mimetypes
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import aiofiles
from services.config_service import MATERIALS_DIR

router = APIRouter(prefix="/api/material")

ALLOWED_TYPES = {
    "image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp",
    "video/mp4", "video/webm", "video/quicktime", "video/x-msvideo",
    "video/x-matroska", "video/mpeg",
}


def _ensure_dir():
    os.makedirs(MATERIALS_DIR, exist_ok=True)


def _file_type(name: str) -> str:
    ext = os.path.splitext(name)[1].lower()
    if ext in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}:
        return "image"
    if ext in {".mp4", ".webm", ".mov", ".avi", ".mkv", ".mpeg", ".mpg"}:
        return "video"
    return "file"


@router.post("/upload")
async def upload_material(file: UploadFile = File(...)):
    _ensure_dir()
    content_type = file.content_type or ""
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {content_type}")

    ext = os.path.splitext(file.filename or "")[1] or mimetypes.guess_extension(content_type) or ""
    filename = f"{uuid.uuid4().hex}{ext}"
    dest = os.path.join(MATERIALS_DIR, filename)

    async with aiofiles.open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            await f.write(chunk)

    stat = os.stat(dest)
    return {
        "success": True,
        "name": filename,
        "original_name": file.filename,
        "path": dest,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "type": _file_type(filename),
        "url": f"/api/material/serve/{filename}",
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
        results.append({
            "name": name,
            "path": full,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "type": _file_type(name),
            "url": f"/api/material/serve/{name}",
        })
    # newest first
    results.sort(key=lambda x: x["mtime"], reverse=True)
    return results


@router.get("/serve/{filename}")
async def serve_material(filename: str):
    _ensure_dir()
    # Prevent path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = os.path.join(MATERIALS_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    mime, _ = mimetypes.guess_type(path)
    return FileResponse(path, media_type=mime or "application/octet-stream")
