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


def make_creator_llm_node(
    creator_llm: Any,
    creator_tools: List[BaseTool],
    system_prompt: Optional[str] = None,
) -> Callable[[HarnessState], Dict[str, Any]]:
    """Return a node function that runs one creator LLM step."""
    bound_llm = creator_llm.bind_tools(creator_tools) if creator_tools else creator_llm

    async def creator_llm_node(state: HarnessState) -> Dict[str, Any]:
        messages = list(state.get("messages", []))

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
