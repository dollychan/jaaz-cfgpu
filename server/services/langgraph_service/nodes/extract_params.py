# server/services/langgraph_service/nodes/extract_params.py
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage


def _last_user_content(messages: List[BaseMessage]) -> str:
    """Return text content of the most recent HumanMessage."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # multimodal: concat text parts
                return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _tag(text: str, tag: str) -> Optional[str]:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _file_ids(text: str, wrapper_tag: str, item_tag: str) -> List[str]:
    wrapper = re.search(rf"<{wrapper_tag}>(.*?)</{wrapper_tag}>", text, re.DOTALL)
    if not wrapper:
        return []
    return re.findall(rf'<{item_tag}[^>]*file_id="([^"]+)"', wrapper.group(1))


def extract_params(state: Dict[str, Any]) -> Dict[str, Any]:
    """Parse generation parameters from the latest user message into state."""
    content = _last_user_content(state.get("messages", []))

    quantity_raw = _tag(content, "quantity")
    try:
        quantity = int(quantity_raw) if quantity_raw else 1
    except ValueError:
        quantity = 1

    duration_raw = _tag(content, "duration")
    try:
        duration: Optional[int] = int(duration_raw) if duration_raw else None
    except ValueError:
        duration = None

    return {
        "quantity": quantity,
        "spec_ratio": _tag(content, "aspect_ratio"),
        "resolution": _tag(content, "resolution"),
        "duration": duration,
        "input_images": _file_ids(content, "input_images", "image"),
        "input_videos": _file_ids(content, "input_videos", "video"),
        "input_audios": _file_ids(content, "input_audios", "audio"),
    }
