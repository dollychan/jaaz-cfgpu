"""
Shared Pydantic field validators for tool schemas.
"""

import json
from typing import Annotated, List, Optional
from pydantic import BeforeValidator


def _parse_str_to_list(v):
    """
    Accept either a proper list or a JSON-encoded string list from the LLM.
    e.g. '["im_abc.png", "im_def.png"]' → ["im_abc.png", "im_def.png"]
    """
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        # Bare single-value string → wrap in list
        return [v]
    return v


# Use this type for Optional list-of-string tool parameters
OptionalStringList = Optional[Annotated[List[str], BeforeValidator(_parse_str_to_list)]]
