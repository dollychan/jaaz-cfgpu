# server/services/langgraph_service/nodes/creator_llm.py
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage
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
    """Remove write_plan tool calls and their ToolMessage results, and remove
    any dangling tool_calls at the end of history that have no ToolMessage result.

    Strict OpenAI-compatible providers reject message histories that:
    1. Reference tools not in the current tool list (write_plan)
    2. End with an AIMessage that has unanswered tool_calls (no ToolMessage follows)
    """
    # Pass 1: remove write_plan calls and their results; capture plan content
    planner_tc_ids: set = set()
    plan_content: Optional[str] = None
    filtered = []
    for msg in messages:
        tc_list = getattr(msg, "tool_calls", None) or []
        planner_ids = {tc["id"] for tc in tc_list if tc.get("name") in _PLANNER_TOOL_NAMES}
        if planner_ids:
            planner_tc_ids.update(planner_ids)
            remaining = [tc for tc in tc_list if tc.get("name") not in _PLANNER_TOOL_NAMES]
            if remaining or getattr(msg, "content", None):
                filtered.append(msg.model_copy(update={"tool_calls": remaining}))
            continue
        tc_id = getattr(msg, "tool_call_id", None)
        if tc_id and tc_id in planner_tc_ids:
            # Capture plan content from the ToolMessage result
            content = getattr(msg, "content", "") or ""
            if content and "<plan>" in content:
                plan_content = content
            continue
        filtered.append(msg)

    # Append plan to the last HumanMessage content (avoids consecutive human messages)
    if plan_content and filtered:
        plan_suffix = f"\n\nExecution plan from planner:\n{plan_content}\n\nPlease execute this plan step by step."
        last_human_idx = None
        for i, m in enumerate(filtered):
            if type(m).__name__ in ("HumanMessage", "human"):
                last_human_idx = i
        if last_human_idx is not None:
            m = filtered[last_human_idx]
            existing = getattr(m, "content", "")
            if isinstance(existing, str):
                filtered[last_human_idx] = m.model_copy(update={"content": existing + plan_suffix})
            elif isinstance(existing, list):
                filtered[last_human_idx] = m.model_copy(update={"content": existing + [{"type": "text", "text": plan_suffix}]})
            print(f"📋 creator_llm: appended plan to last HumanMessage (idx={last_human_idx})")

    # Pass 2: remove dangling tool_calls (AIMessage with tool_calls but no following ToolMessage)
    answered_tc_ids: set = {
        getattr(m, "tool_call_id", None)
        for m in filtered
        if getattr(m, "tool_call_id", None)
    }
    pass2 = []
    for msg in filtered:
        tc_list = getattr(msg, "tool_calls", None) or []
        if tc_list:
            live = [tc for tc in tc_list if tc.get("id") in answered_tc_ids]
            if live != tc_list:
                dropped = [tc["name"] for tc in tc_list if tc.get("id") not in answered_tc_ids]
                print(f"🧹 creator_llm: dropping dangling tool_calls with no result: {dropped}")
            if live:
                pass2.append(msg.model_copy(update={"tool_calls": live}))
            elif getattr(msg, "content", None):
                pass2.append(msg.model_copy(update={"tool_calls": []}))
            # else: drop entirely (no content, no valid tool_calls)
        else:
            pass2.append(msg)

    # Pass 3: merge consecutive same-role messages — strict OpenAI-compatible providers
    # reject histories with two AI (or two Human) messages in a row.
    from langchain_core.messages import AIMessage as _AIMessage, HumanMessage as _HumanMessage
    result = []
    for msg in pass2:
        if result and isinstance(result[-1], _AIMessage) and isinstance(msg, _AIMessage):
            prev = result[-1]
            prev_tc = getattr(prev, "tool_calls", None) or []
            cur_tc = getattr(msg, "tool_calls", None) or []
            if not prev_tc:
                print(f"🧹 creator_llm: merging consecutive AI messages (dropping text-only predecessor)")
                result[-1] = msg
            elif not cur_tc:
                print(f"🧹 creator_llm: merging consecutive AI messages (dropping text-only successor)")
            else:
                result.append(msg)
        elif result and isinstance(result[-1], _HumanMessage) and isinstance(msg, _HumanMessage):
            # Merge consecutive human messages by appending content
            prev = result[-1]
            prev_content = getattr(prev, "content", "")
            cur_content = getattr(msg, "content", "")
            if isinstance(prev_content, str) and isinstance(cur_content, str):
                print(f"🧹 creator_llm: merging consecutive Human messages")
                result[-1] = prev.model_copy(update={"content": prev_content + "\n" + cur_content})
            else:
                result.append(msg)
        else:
            result.append(msg)
    return result


def make_creator_llm_node(
    creator_llm: Any,
    creator_tools: List[BaseTool],
    system_prompt: Optional[str] = None,
) -> Callable[[HarnessState], Dict[str, Any]]:
    """Return a node function that runs one creator LLM step."""
    bound_llm = creator_llm.bind_tools(creator_tools) if creator_tools else creator_llm

    async def creator_llm_node(state: HarnessState) -> Dict[str, Any]:
        raw_messages = list(state.get("messages", []))
        messages = _strip_planner_messages(raw_messages)

        print(f"🎨 creator_llm: {len(raw_messages)} raw msgs → {len(messages)} after strip")
        for i, m in enumerate(messages):
            tc = getattr(m, "tool_calls", None)
            tc_id = getattr(m, "tool_call_id", None)
            role = getattr(m, "type", type(m).__name__)
            content_preview = str(getattr(m, "content", ""))[:80]
            print(f"  [{i}] {role} | tc={[t.get('name') for t in (tc or [])] if tc else None} | tc_id={tc_id} | {content_preview!r}")

        # Build system messages: agent system prompt + harness context
        system_messages = []
        if system_prompt:
            system_messages.append(SystemMessage(content=system_prompt))
        system_messages.append(SystemMessage(content=_build_harness_context(state)))

        augmented = system_messages + messages

        response = await bound_llm.ainvoke(augmented)

        # Extract valid tool_calls (filter phantom entries with empty name/id)
        tool_calls = getattr(response, "tool_calls", []) or []
        valid_tool_calls = [tc for tc in tool_calls if tc.get("name") and tc.get("id")]
        phantom_count = len(tool_calls) - len(valid_tool_calls)
        if phantom_count:
            print(f"🧹 creator_llm: dropping {phantom_count} phantom tool_call(s) with empty name/id")
            response = response.model_copy(update={"tool_calls": valid_tool_calls})

        pending = [
            {"id": tc["id"], "name": tc["name"], "args": tc["args"]}
            for tc in valid_tool_calls
        ]

        return {
            "messages": [response],
            "pending_tool_calls": pending,
            "error_context": None,   # reset each iteration
        }

    return creator_llm_node
