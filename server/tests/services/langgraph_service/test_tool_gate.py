import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from services.langgraph_service.nodes.tool_gate import make_tool_gate_node, DEFAULT_TOOL_ALLOWLIST


def _state(pending, approval_given=False, approved_tool_name=None):
    return {
        "session_id": "sess-1",
        "pending_tool_calls": pending,
        "approval_given": approval_given,
        "approved_tool_name": approved_tool_name,
    }


@pytest.mark.asyncio
async def test_allowlisted_tool_auto_approved():
    allowlisted_tool = next(iter(DEFAULT_TOOL_ALLOWLIST))
    pending = [{"id": "tc1", "name": allowlisted_tool, "args": {"prompt": "cat"}}]
    mock_ws = AsyncMock()
    node = make_tool_gate_node(websocket_service=mock_ws)
    result = await node(_state(pending))
    assert result["approved_tool_calls"] == pending
    mock_ws.assert_not_called()


@pytest.mark.asyncio
async def test_same_tool_name_auto_approved_after_first():
    pending = [{"id": "tc2", "name": "generate_video_by_veo3_fast_jaaz", "args": {}}]
    mock_ws = AsyncMock()
    node = make_tool_gate_node(websocket_service=mock_ws)
    result = await node(_state(pending, approval_given=True, approved_tool_name="generate_video_by_veo3_fast_jaaz"))
    assert result["approved_tool_calls"] == pending
    mock_ws.assert_not_called()


@pytest.mark.asyncio
async def test_unapproved_tool_sends_ws_event_and_awaits():
    pending = [{"id": "tc3", "name": "generate_video_by_veo3_fast_jaaz", "args": {"prompt": "ocean"}}]
    mock_ws = AsyncMock()

    with patch("services.langgraph_service.nodes.tool_gate.batch_approval_manager") as mock_mgr:
        mock_mgr.request_approval = AsyncMock(return_value=pending)
        node = make_tool_gate_node(websocket_service=mock_ws)
        result = await node(_state(pending, approval_given=False))

    mock_ws.assert_called_once()
    ws_call_args = mock_ws.call_args[0]
    assert ws_call_args[1]["type"] == "tool_approval_request"
    assert result["approval_given"] is True
    assert result["approved_tool_name"] == "generate_video_by_veo3_fast_jaaz"
