# server/services/langgraph_service/nodes/planner_llm.py
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from services.langgraph_service.state import HarnessState


def make_planner_llm_node(
    planner_llm: Any,
    planner_tools: List[BaseTool],
    system_prompt: Optional[str] = None,
) -> Callable[[HarnessState], Dict[str, Any]]:
    """Return a node function that runs one planner LLM step."""
    bound_llm = planner_llm.bind_tools(planner_tools)

    async def planner_llm_node(state: HarnessState) -> Dict[str, Any]:
        messages = list(state.get("messages", []))
        if system_prompt:
            messages = [SystemMessage(content=system_prompt)] + messages
        response = await bound_llm.ainvoke(messages)
        return {"messages": [response]}

    return planner_llm_node
