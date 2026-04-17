# server/services/langgraph_service/nodes/validate_plan.py
from __future__ import annotations
from typing import Any, Dict
from langchain_core.messages import ToolMessage
from services.langgraph_service.state import HarnessState, ErrorContext


def validate_plan(state: HarnessState) -> Dict[str, Any]:
    """Check that the planner called write_plan before completing.

    CURRENT BEHAVIOR:
    - Only verifies that write_plan was called (binary pass/fail).
    - The outgoing edge is always fixed to creator_llm regardless of the result.
    - plan_validated=False sets an error_context but does not trigger a retry.

    TODO — Future: Upgrade to a plan evaluation agent
    This node is a placeholder for a richer plan evaluation step. Planned improvements:

    1. Plan scoring: invoke an LLM judge to score the plan on dimensions such as
       completeness, step ordering, prompt quality, and alignment with user intent.

    2. Conditional routing: replace the fixed edge with a conditional edge so that:
       - score >= threshold  →  creator_llm  (proceed with execution)
       - score <  threshold  →  planner_llm  (ask planner to revise)
       - retry_count > limit →  creator_llm  (fall back gracefully to avoid infinite loop)

    3. Feedback injection: when routing back to planner_llm, inject the evaluator's
       critique as a HumanMessage so the planner knows what to improve.

    4. Metrics: emit plan score and retry count as LangSmith trace metadata for
       offline analysis and prompt iteration.
    """
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
