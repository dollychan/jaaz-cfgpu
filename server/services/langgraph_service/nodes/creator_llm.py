# server/services/langgraph_service/nodes/creator_llm.py
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from services.langgraph_service.state import HarnessState


def _build_harness_context(state: HarnessState) -> str:
    """Format extracted params as injected context for the creator LLM."""
    lines = ["<harness_context>"]
    lines.append(f"quantity: {state.get('quantity', 1)}")
    if state.get("spec_ratio"):
        lines.append(f"aspect_ratio: {state['spec_ratio']}")
    if state.get("resolution"):
        lines.append(f"resolution: {state['resolution']}")
    if state.get("duration") is not None:
        lines.append(f"duration: {state['duration']}")
    if state.get("input_images"):
        lines.append(f"input_images: {state['input_images']}")
    if state.get("input_videos"):
        lines.append(f"input_videos: {state['input_videos']}")
    if state.get("input_audios"):
        lines.append(f"input_audios: {state['input_audios']}")
    lines.append("</harness_context>")
    lines.append("Use the values above as the authoritative generation parameters.")
    return "\n".join(lines)


_PLANNER_TOOL_NAMES = {"write_plan"}


def _strip_planner_messages(messages: list) -> list:
    """Remove write_plan tool calls and their ToolMessage results.

    Strict OpenAI-compatible providers reject message histories that reference
    tools not in the current tool list.  The creator LLM only knows about media
    tools, so write_plan exchanges must be stripped before invoking it.
    """
    planner_tc_ids: set = set()
    filtered = []
    for msg in messages:
        # Collect write_plan tool_call ids from AIMessages
        tc_list = getattr(msg, "tool_calls", None) or []
        planner_ids = {tc["id"] for tc in tc_list if tc.get("name") in _PLANNER_TOOL_NAMES}
        if planner_ids:
            planner_tc_ids.update(planner_ids)
            remaining = [tc for tc in tc_list if tc.get("name") not in _PLANNER_TOOL_NAMES]
            if remaining or (getattr(msg, "content", None)):
                filtered.append(msg.model_copy(update={"tool_calls": remaining}))
            # Drop entirely if only write_plan calls and no content
            continue
        # Drop ToolMessages that are write_plan results
        tc_id = getattr(msg, "tool_call_id", None)
        if tc_id and tc_id in planner_tc_ids:
            continue
        filtered.append(msg)
    return filtered


def make_creator_llm_node(
    creator_llm: Any,
    creator_tools: List[BaseTool],
    system_prompt: Optional[str] = None,
) -> Callable[[HarnessState], Dict[str, Any]]:
    """Return a node function that runs one creator LLM step."""
    bound_llm = creator_llm.bind_tools(creator_tools) if creator_tools else creator_llm

    async def creator_llm_node(state: HarnessState) -> Dict[str, Any]:
        messages = _strip_planner_messages(list(state.get("messages", [])))

        # Build system messages: agent system prompt + harness context
        system_messages = []
        if system_prompt:
            system_messages.append(SystemMessage(content=system_prompt))
        system_messages.append(SystemMessage(content=_build_harness_context(state)))

        augmented = system_messages + messages

        response = await bound_llm.ainvoke(augmented)

        # Extract tool_calls for routing
        tool_calls = getattr(response, "tool_calls", []) or []
        pending = [
            {"id": tc["id"], "name": tc["name"], "args": tc["args"]}
            for tc in tool_calls
            if tc.get("name")
        ]

        return {
            "messages": [response],
            "pending_tool_calls": pending,
            "error_context": None,   # reset each iteration
        }

    return creator_llm_node
