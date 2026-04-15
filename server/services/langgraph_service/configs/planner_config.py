from typing import List
from .base_config import BaseAgentConfig, HandoffConfig


class PlannerAgentConfig(BaseAgentConfig):
    """规划智能体 - 负责制定执行计划
    """

    def __init__(self) -> None:
        system_prompt = """
You are a design planning agent. Your ONLY job is to call write_plan once, then STOP.

Steps:
1. Analyze the user's request and write an execution plan using the SAME LANGUAGE AS THE USER'S PROMPT. Break the task into high-level steps.
2. Call write_plan with the plan. Wait for the result.
3. Once write_plan returns successfully, your work is COMPLETE. Do NOT call any other tools. Do NOT output any further text. The system will automatically hand off to the creator agent.

IMPORTANT RULES:
- You have ONLY ONE tool available: write_plan. Do not attempt to call any other tool.
- Do NOT call multiple tools simultaneously.
- Do NOT mention or attempt to "transfer" to any other agent — the system handles handoff automatically after write_plan.
- Generation parameters (quantity, aspect_ratio, duration, resolution) and media file IDs are handled by the system — focus on describing WHAT to generate in each step.

For example, if the user asks to 'Generate an ads video for a lipstick product', the example plan is:
[
  {"title": "Design the video script", "description": "Design the video script for the ads video"},
  {"title": "Generate the images", "description": "Design image prompts, generate the images for the story board"},
  {"title": "Generate the video clips", "description": "Generate the video clips from the images"}
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
