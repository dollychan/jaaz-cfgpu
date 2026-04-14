# server/services/langgraph_service/harness_graph.py
from __future__ import annotations
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from services.langgraph_service.state import HarnessState
from services.langgraph_service.nodes import (
    extract_params,
    make_planner_llm_node,
    validate_plan,
    make_creator_llm_node,
    make_tool_gate_node,
    error_handler,
    classify_tool_error,
)


# ── Routing functions ────────────────────────────────────────────────────────

def _route_after_extract(state: HarnessState) -> str:
    return "planner_llm" if state.get("use_planner") else "creator_llm"


def _route_planner(state: HarnessState) -> str:
    """After planner LLM: go to tool execution or validate (if no tool calls)."""
    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    if last and getattr(last, "tool_calls", None):
        return "planner_tools"
    return "validate_plan"


def _route_creator(state: HarnessState) -> str:
    """After creator LLM: go to tool_gate (has tool calls) or END (finished)."""
    if state.get("hard_stop"):
        return END
    pending = state.get("pending_tool_calls", [])
    return "tool_gate" if pending else END


def _route_after_creator_tools(state: HarnessState) -> str:
    """After tool execution: check for hard errors or loop back to creator."""
    if state.get("hard_stop"):
        return END
    ctx = state.get("error_context")
    if ctx:
        return "error_handler"
    return "creator_llm"


def _route_after_error(state: HarnessState) -> str:
    if state.get("hard_stop"):
        return END
    # SERVICE_TIMEOUT retry: pending/approved tool calls still set → re-gate
    if state.get("pending_tool_calls") or state.get("approved_tool_calls"):
        return "tool_gate"
    return "creator_llm"


# ── Graph builder ────────────────────────────────────────────────────────────

def build_harness_graph(
    planner_llm: Optional[Any],
    creator_llm: Any,
    planner_tools: List[BaseTool],
    creator_tools: List[BaseTool],
    websocket_service: Callable[..., Coroutine[Any, Any, None]],
    tool_allowlist: Optional[Set[str]] = None,
    planner_system_prompt: Optional[str] = None,
    creator_system_prompt: Optional[str] = None,
) -> Any:
    """Build and compile the harness StateGraph.

    Args:
        planner_llm: LLM for the planner (None if single-agent / creator-only mode).
        creator_llm: LLM for the creator.
        planner_tools: Tools for planner (write_plan only).
        creator_tools: Image/video generation tools.
        websocket_service: Callable for sending WS events (session_id, event_dict).
        tool_allowlist: Set of tool names that bypass human approval.
            Defaults to DEFAULT_TOOL_ALLOWLIST in tool_gate.py.
        planner_system_prompt: System prompt for planner LLM.
        creator_system_prompt: System prompt for creator LLM.

    Returns:
        Compiled LangGraph graph ready for .astream().
    """
    graph = StateGraph(HarnessState)

    # ── creator_tools node (ToolNode wrapped with error classification) ──────
    async def creator_tools_node(state: HarnessState) -> Dict[str, Any]:
        """Execute approved tool calls and classify any errors."""
        approved = state.get("approved_tool_calls", [])
        if not approved:
            return {"pending_tool_calls": [], "approved_tool_calls": [], "error_context": None}

        messages = list(state.get("messages", []))
        last_ai = messages[-1] if messages else None

        # Patch the last AIMessage's tool_calls with the (potentially user-modified) approved calls
        if last_ai and hasattr(last_ai, "tool_calls"):
            patched = last_ai.model_copy(
                update={"tool_calls": [
                    {"id": tc["id"], "name": tc["name"], "args": tc["args"]}
                    for tc in approved
                ]}
            )
            patched_messages = messages[:-1] + [patched]
        else:
            patched_messages = messages

        patched_state = {**state, "messages": patched_messages, "approved_tool_calls": approved}

        tool_node = ToolNode(creator_tools)
        result = await tool_node.ainvoke(patched_state)

        # Classify errors from ToolMessages
        new_messages = result.get("messages", [])
        error_ctx = None
        for msg in new_messages:
            if isinstance(msg, ToolMessage):
                content_str = str(msg.content)
                err_type = classify_tool_error(content_str)
                if err_type != "OTHER":
                    error_ctx = {
                        "type": err_type,
                        "tool_name": msg.name,
                        "message": content_str,
                        "retry_count": state.get("retry_count", 0),
                    }
                    break
                # Also catch generic generation failure messages
                if "generation failed:" in content_str.lower():
                    error_ctx = {
                        "type": "OTHER",
                        "tool_name": msg.name,
                        "message": content_str,
                        "retry_count": state.get("retry_count", 0),
                    }
                    break

        return {
            **result,
            "pending_tool_calls": [],
            "approved_tool_calls": [],
            "error_context": error_ctx,
        }

    # ── Nodes ────────────────────────────────────────────────────────────────
    graph.add_node("extract_params", extract_params)

    if planner_llm is not None:
        graph.add_node("planner_llm", make_planner_llm_node(planner_llm, planner_tools, planner_system_prompt))
        graph.add_node("planner_tools", ToolNode(planner_tools))
        graph.add_node("validate_plan", validate_plan)

    graph.add_node("creator_llm", make_creator_llm_node(creator_llm, creator_tools, creator_system_prompt))
    graph.add_node("tool_gate", make_tool_gate_node(websocket_service, tool_allowlist))
    graph.add_node("creator_tools", creator_tools_node)
    graph.add_node("error_handler", error_handler)

    # ── Edges ────────────────────────────────────────────────────────────────
    graph.add_edge(START, "extract_params")
    graph.add_conditional_edges("extract_params", _route_after_extract)

    if planner_llm is not None:
        graph.add_conditional_edges("planner_llm", _route_planner)
        graph.add_edge("planner_tools", "planner_llm")   # loop until planner done
        graph.add_edge("validate_plan", "creator_llm")

    graph.add_conditional_edges("creator_llm", _route_creator)
    graph.add_edge("tool_gate", "creator_tools")
    graph.add_conditional_edges("creator_tools", _route_after_creator_tools)
    graph.add_conditional_edges("error_handler", _route_after_error)

    return graph.compile()
