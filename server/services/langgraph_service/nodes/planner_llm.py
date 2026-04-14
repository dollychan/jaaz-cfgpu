# server/services/langgraph_service/nodes/planner_llm.py
from __future__ import annotations
from typing import Any, Callable, Dict, List
from langchain_core.tools import BaseTool
from services.langgraph_service.state import HarnessState


def make_planner_llm_node(
    planner_llm: Any,
    planner_tools: List[BaseTool],
) -> Callable[[HarnessState], Dict[str, Any]]:
    """Return a node function that runs one planner LLM step."""
    bound_llm = planner_llm.bind_tools(planner_tools)

    async def planner_llm_node(state: HarnessState) -> Dict[str, Any]:
        messages = list(state.get("messages", []))
        response = await bound_llm.ainvoke(messages)
        return {"messages": [response]}

    return planner_llm_node
