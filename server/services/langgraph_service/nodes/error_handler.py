# server/services/langgraph_service/nodes/error_handler.py
from __future__ import annotations
from typing import Any, Dict
from langchain_core.messages import AIMessage
from services.langgraph_service.state import HarnessState, ErrorContext


def classify_tool_error(tool_result: str) -> str:
    """Return error type string from a tool result message."""
    lowered = tool_result.lower()
    if "content policy violation" in lowered or "stop. do not retry" in lowered:
        return "CONTENT_POLICY_VIOLATION"
    if "timed out" in lowered or "timeout" in lowered or "504" in lowered:
        return "SERVICE_TIMEOUT"
    if "balance" in lowered or "quota exceeded" in lowered or "402" in lowered:
        return "INSUFFICIENT_BALANCE"
    return "OTHER"


def error_handler(state: HarnessState) -> Dict[str, Any]:
    """Classify error and decide next action (encoded in hard_stop / retry_count)."""
    ctx: ErrorContext = state.get("error_context") or {}
    error_type = ctx.get("type", "OTHER")
    retry_count = state.get("retry_count", 0)

    if error_type == "CONTENT_POLICY_VIOLATION":
        msg = AIMessage(content="⚠️ Content policy violation — generation stopped. Please revise your request.")
        return {"messages": [msg], "hard_stop": True}

    if error_type == "INSUFFICIENT_BALANCE":
        msg = AIMessage(content="⚠️ Insufficient balance or quota exceeded — generation stopped. Please top up and retry.")
        return {"messages": [msg], "hard_stop": True}

    if error_type == "SERVICE_TIMEOUT":
        if retry_count < 2:
            print(f"⏳ Timeout retry {retry_count + 1}/2 for {ctx.get('tool_name')}")
            return {
                "retry_count": retry_count + 1,
                "hard_stop": False,
            }
        # Max retries exceeded — skip and continue
        print(f"❌ Timeout max retries exceeded for {ctx.get('tool_name')}, skipping")
        return {"retry_count": 0, "hard_stop": False, "pending_tool_calls": [], "approved_tool_calls": []}

    if error_type == "PLANNER_NO_PLAN":
        # Non-fatal: proceed to creator without validated plan
        return {"hard_stop": False}

    # OTHER — skip step, continue
    return {"retry_count": 0, "hard_stop": False, "pending_tool_calls": [], "approved_tool_calls": []}
