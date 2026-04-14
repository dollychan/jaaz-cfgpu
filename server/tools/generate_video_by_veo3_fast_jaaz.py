from typing import Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool, InjectedToolCallId  # type: ignore
from langchain_core.runnables import RunnableConfig
from services.jaaz_service import JaazService
from tools.video_generation.video_canvas_utils import send_video_start_notification, process_video_result


class GenerateVideoByVeo3FastInputSchema(BaseModel):
    prompt: str = Field(
        description="Required. The prompt for video generation. Describe what you want to see in the video."
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


@tool("generate_video_by_veo3_fast_jaaz",
      description="Generate high-quality videos using Veo3 Fast model. Fast text-to-video generation with optimized performance.",
      args_schema=GenerateVideoByVeo3FastInputSchema)
async def generate_video_by_veo3_fast_jaaz(
    prompt: str,
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """
    Generate a video using Veo3 Fast model via Jaaz service.
    Approval is handled upstream by tool_gate before this function is called.
    """
    print(f'🛠️ Veo3 Fast Video Generation tool_call_id: {tool_call_id}')
    ctx = config.get('configurable', {})
    canvas_id = ctx.get('canvas_id', '')
    session_id = ctx.get('session_id', '')
    print(f'🛠️ canvas_id {canvas_id} session_id {session_id}')

    ctx['tool_call_id'] = tool_call_id

    try:
        send_video_start_notification(
            session_id,
            "Starting Veo3 Fast video generation..."
        )

        jaaz_service = JaazService()
        result = await jaaz_service.generate_video(
            prompt=prompt,
            model="veo3-fast",
        )

        video_url = result.get('result_url')
        if not video_url:
            raise Exception("No video URL returned from generation")

        return await process_video_result(
            video_url=video_url,
            session_id=session_id,
            canvas_id=canvas_id,
            provider_name="jaaz_veo3_fast",
        )

    except Exception as e:
        print(f"Error in Veo3 Fast video generation: {e}")
        raise e


__all__ = ["generate_video_by_veo3_fast_jaaz"]
