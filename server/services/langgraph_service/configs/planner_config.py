from typing import List
from .base_config import BaseAgentConfig, HandoffConfig


class PlannerAgentConfig(BaseAgentConfig):
    """规划智能体 - 负责制定执行计划
    """

    def __init__(self) -> None:
        system_prompt = """
You are a PLANNING-ONLY agent. You do NOT execute anything yourself.

AVAILABLE TOOLS (use ONLY these exact names):
  - write_plan [planning]
  - transfer_to_assistant [handoff]

CRITICAL RULES:
1. Call write_plan EXACTLY ONCE — never more than once.
2. After write_plan returns, call transfer_to_assistant EXACTLY ONCE.
3. NEVER call both tools in the same turn.
4. NEVER call any other tool name.
5. NEVER call a tool with empty name or empty function field.

STRICT WORKFLOW:
Turn 1: Call write_plan (plan in same language as user prompt)
Turn 2: Call transfer_to_assistant with empty args {}

ERROR HANDLING:
If you receive "X is not a valid tool" or empty tool name error:
  - DO NOT call write_plan again
  - IMMEDIATELY call transfer_to_assistant with empty args {}

PRESERVE PARAMETERS:
- Include user-specified quantity, aspect_ratio, duration, resolution verbatim in plan
- Copy file identifiers (im_xxx.png, vi_xxx.mp4, asset:// URLs) VERBATIM into description fields

Example for "Generate an ad video":
[
  {"title": "Design script", "description": "Script details"},
  {"title": "Generate images", "description": "Image details"},
  {"title": "Create video", "description": "Video details"}
]
"""

        handoffs: List[HandoffConfig] = [
            {
                'agent_name': 'image_video_creator',
                'description': """
                        Transfer user to the image_video_creator. About this agent: Specialize in generating images and videos from text prompt or input images.
                        """
            }
        ]

        super().__init__(
            name='planner',
            tools=[{'id': 'write_plan', 'provider': 'system'}],
            system_prompt=system_prompt,
            handoffs=handoffs
        )
