# server/services/langgraph_service/nodes/validate_plan.py
from __future__ import annotations
from typing import Any, Dict
from langchain_core.messages import ToolMessage
from services.langgraph_service.state import HarnessState, ErrorContext


def validate_plan(state: HarnessState) -> Dict[str, Any]:
    """Check that the planner called write_plan before completing."""
    messages = state.get("messages", [])
    plan_found = any(
        isinstance(m, ToolMessage) and m.name == "write_plan"
        for m in messages
    )
    if plan_found:
        return {"plan_validated": True}

    error: ErrorContext = {
        "type": "PLANNER_NO_PLAN",
        "message": "Planner did not call write_plan before completing.",
        "retry_count": 0,
    }
    print("⚠️  validate_plan: write_plan result not found — proceeding to creator without validated plan")
    return {"plan_validated": False, "error_context": error}
