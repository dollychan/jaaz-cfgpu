# server/services/langgraph_service/state.py
from __future__ import annotations
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from models.tool_model import ToolInfoJson


class ErrorContext(TypedDict, total=False):
    type: str        # CONTENT_POLICY_VIOLATION | SERVICE_TIMEOUT | INSUFFICIENT_BALANCE | PLANNER_NO_PLAN | OTHER
    tool_name: str
    message: str
    retry_count: int


class HarnessState(TypedDict, total=False):
    # ── Input (set once at entry) ────────────────────────────────────────────
    messages: Annotated[list, add_messages]
    canvas_id: str
    session_id: str
    tool_list: List[ToolInfoJson]
    system_prompt: Optional[str]
    use_planner: bool              # True when tool_list contains a text tool

    # ── Extracted generation parameters (set by extract_params) ─────────────
    quantity: int                  # default 1
    spec_ratio: Optional[str]      # aspect_ratio value
    resolution: Optional[str]      # video resolution e.g. "720p"
    duration: Optional[int]        # video duration in seconds
    input_images: List[str]        # file_ids from latest user message
    input_videos: List[str]
    input_audios: List[str]

    # ── Planner state ────────────────────────────────────────────────────────
    plan_validated: bool           # write_plan tool was called and result found

    # ── Creator / tool execution state ──────────────────────────────────────
    pending_tool_calls: List[Dict[str, Any]]   # from latest creator LLM response
    approved_tool_calls: List[Dict[str, Any]]  # after tool_gate
    approval_given: bool           # True once user has approved at least once
    approved_tool_name: Optional[str]          # tool name approved; auto-approve same name

    # ── Error handling ───────────────────────────────────────────────────────
    error_context: Optional[ErrorContext]
    retry_count: int
    hard_stop: bool                # True → skip to END immediately
