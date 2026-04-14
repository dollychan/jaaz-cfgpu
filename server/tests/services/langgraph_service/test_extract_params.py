import pytest
from langchain_core.messages import HumanMessage, AIMessage
from services.langgraph_service.nodes.extract_params import extract_params


def _state(content: str) -> dict:
    return {
        "messages": [HumanMessage(content=content)],
        "tool_list": [],
    }


def test_extracts_quantity():
    result = extract_params(_state("Generate images <quantity>5</quantity>"))
    assert result["quantity"] == 5


def test_defaults_quantity_to_1():
    result = extract_params(_state("Generate an image of a cat"))
    assert result["quantity"] == 1


def test_extracts_aspect_ratio():
    result = extract_params(_state("Make a video <aspect_ratio>16:9</aspect_ratio>"))
    assert result["spec_ratio"] == "16:9"


def test_extracts_duration():
    result = extract_params(_state("Create a video <duration>10</duration>"))
    assert result["duration"] == 10


def test_extracts_resolution():
    result = extract_params(_state("Generate <resolution>720p</resolution>"))
    assert result["resolution"] == "720p"


def test_extracts_input_images():
    content = """Generate
    <input_images>
      <image file_id="im_abc123.png"/>
      <image file_id="im_def456.png"/>
    </input_images>"""
    result = extract_params(_state(content))
    assert result["input_images"] == ["im_abc123.png", "im_def456.png"]


def test_extracts_input_videos():
    content = '<input_videos><video file_id="vi_xyz.mp4"/></input_videos>'
    result = extract_params(_state(content))
    assert result["input_videos"] == ["vi_xyz.mp4"]


def test_extracts_input_audios():
    content = '<input_audios><audio file_id="au_zzz.mp3"/></input_audios>'
    result = extract_params(_state(content))
    assert result["input_audios"] == ["au_zzz.mp3"]


def test_uses_latest_message_only():
    """Params from earlier messages must be ignored; only latest user message counts."""
    state = {
        "messages": [
            HumanMessage(content="Make 3 images <quantity>3</quantity>"),
            AIMessage(content="Done"),
            HumanMessage(content="Now make 1 image <quantity>1</quantity>"),
        ],
        "tool_list": [],
    }
    result = extract_params(state)
    assert result["quantity"] == 1


def test_asset_url_in_input_images():
    content = 'Use asset <input_images><image file_id="asset://asset-abc"/></input_images>'
    result = extract_params(_state(content))
    assert result["input_images"] == ["asset://asset-abc"]
