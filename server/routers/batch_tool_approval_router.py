# server/routers/batch_tool_approval_router.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List
from services.websocket_service import send_to_websocket
from services.batch_approval_manager import batch_approval_manager

router = APIRouter(prefix="/api")


class BatchToolApprovalRequest(BaseModel):
    session_id: str
    batch_id: str
    approved: bool
    tool_calls: List[Dict[str, Any]] = []  # may contain user-modified payloads


@router.post("/batch_tool_approval")
async def handle_batch_tool_approval(request: BatchToolApprovalRequest):
    """Handle frontend approval/rejection of a batch of tool calls."""
    if request.approved:
        success = batch_approval_manager.approve_batch(request.batch_id, request.tool_calls)
        if not success:
            raise HTTPException(status_code=404, detail="Batch not found or already processed")
        await send_to_websocket(request.session_id, {
            'type': 'tool_batch_approved',
            'batch_id': request.batch_id,
        })
    else:
        success = batch_approval_manager.reject_batch(request.batch_id)
        if not success:
            raise HTTPException(status_code=404, detail="Batch not found or already processed")
        await send_to_websocket(request.session_id, {
            'type': 'tool_batch_rejected',
            'batch_id': request.batch_id,
        })
    return {"status": "success"}
