# server/routers/batch_tool_approval_router.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List
from services.websocket_service import send_to_websocket
from services.batch_approval_manager import batch_approval_manager
from services.db_service import db_service

router = APIRouter(prefix="/api")


class BatchToolApprovalRequest(BaseModel):
    session_id: str
    batch_id: str
    approved: bool
    tool_calls: List[Dict[str, Any]] = []  # may contain user-modified payloads


def _normalize_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize tool_calls from frontend format to internal format.

    Frontend sends: {"id": ..., "name": ..., "arguments": {...}}
    Internal format: {"id": ..., "name": ..., "args": {...}}
    """
    normalized = []
    for tc in tool_calls:
        tc_id = tc.get("id")
        tc_name = tc.get("name")
        if not tc_id or not tc_name:
            raise ValueError(f"Tool call missing required 'id' or 'name' field: {tc}")
        # Accept both "args" and "arguments" from frontend
        tc_args = tc.get("args") or tc.get("arguments") or {}
        normalized.append({"id": tc_id, "name": tc_name, "args": tc_args})
    return normalized


@router.post("/batch_tool_approval")
async def handle_batch_tool_approval(request: BatchToolApprovalRequest):
    """Handle frontend approval/rejection of a batch of tool calls."""
    if request.approved:
        if not request.tool_calls:
            raise HTTPException(
                status_code=422,
                detail="approved=true requires at least one tool_call with 'id' and 'name'"
            )
        try:
            normalized = _normalize_tool_calls(request.tool_calls)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        success = batch_approval_manager.approve_batch(request.batch_id, normalized)
        if not success:
            raise HTTPException(status_code=404, detail="Batch not found or already processed")
        # Persist edited args to DB so chat history reflects the approved values
        await db_service.update_last_assistant_tool_call_args(request.session_id, normalized)
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
