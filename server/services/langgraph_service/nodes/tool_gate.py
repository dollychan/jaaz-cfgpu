# server/services/langgraph_service/nodes/tool_gate.py
from __future__ import annotations
import uuid
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set
from services.batch_approval_manager import batch_approval_manager
from services.langgraph_service.state import HarnessState

# Tools that are auto-approved without user confirmation.
# Extend via settings.json in future.
DEFAULT_TOOL_ALLOWLIST: Set[str] = {
    "generate_image_by_gpt_image_1_jaaz",
    "generate_image_by_imagen_4_jaaz",
    "generate_image_by_imagen_4_replicate",
    "generate_image_by_flux_kontext_pro_jaaz",
    "generate_image_by_flux_kontext_pro_replicate",
    "generate_image_by_flux_kontext_max_jaaz",
    "generate_image_by_flux_kontext_max_replicate",
    "generate_image_by_ideogram3_bal_jaaz",
    "generate_image_by_recraft_v3_jaaz",
    "generate_image_by_recraft_v3_replicate",
    "generate_image_by_doubao_seedream_3_jaaz",
    "generate_image_by_doubao_seedream_3_volces",
    "generate_image_by_doubao_seedream_4_0_cfgpu",
    "generate_image_by_doubao_seedream_4_5_cfgpu",
    "generate_image_by_doubao_seedream_5_0_cfgpu",
    "generate_image_by_midjourney_jaaz",
    "edit_image_by_doubao_seededit_3_volces",
    "generate_video_by_seedance_v1_jaaz",
    "generate_video_by_seedance_v1_pro_volces",
    "generate_video_by_seedance_v1_lite_t2v",
    "generate_video_by_seedance_v1_lite_i2v",
    "generate_video_by_cfgpu_seedance_2_0",
    "generate_video_by_cfgpu_seedance_2_0_fast",
    "generate_video_by_kling_v2_jaaz",
    "generate_video_by_hailuo_02_jaaz",
    # generate_video_by_veo3_fast_jaaz intentionally NOT here → requires approval
}


def make_tool_gate_node(
    websocket_service: Callable[..., Coroutine[Any, Any, None]],
    allowlist: Optional[Set[str]] = None,
) -> Callable[[HarnessState], Coroutine[Any, Any, Dict[str, Any]]]:
    """Return an async node that gates tool calls through approval."""
    effective_allowlist = allowlist if allowlist is not None else DEFAULT_TOOL_ALLOWLIST

    async def tool_gate_node(state: HarnessState) -> Dict[str, Any]:
        pending: List[Dict[str, Any]] = state.get("pending_tool_calls", [])
        if not pending:
            return {"approved_tool_calls": []}

        session_id = state.get("session_id", "")
        approval_given = state.get("approval_given", False)
        approved_tool_name = state.get("approved_tool_name")

        def _auto_approve(tc: Dict[str, Any]) -> bool:
            name = tc.get("name", "")
            if name in effective_allowlist:
                return True
            # Same tool name as a previously approved call — auto-approve
            if approval_given and name == approved_tool_name:
                return True
            return False

        if all(_auto_approve(tc) for tc in pending):
            return {"approved_tool_calls": pending}

        # Need user approval — send event and await
        batch_id = str(uuid.uuid4())
        await websocket_service(session_id, {
            "type": "tool_approval_request",
            "batch_id": batch_id,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "name": tc["name"],
                    "arguments": tc["args"],
                }
                for tc in pending
            ],
        })

        approved = await batch_approval_manager.request_approval(batch_id, pending)

        updates: Dict[str, Any] = {"approved_tool_calls": approved}
        if approved:
            updates["approval_given"] = True
            updates["approved_tool_name"] = approved[0].get("name")

        return updates

    return tool_gate_node
