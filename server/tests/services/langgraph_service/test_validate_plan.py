import pytest
from langchain_core.messages import AIMessage, ToolMessage
from services.langgraph_service.nodes.validate_plan import validate_plan


def _state_with_plan():
    """State where planner has called write_plan and received result."""
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"id": "tc1", "name": "write_plan", "args": {"steps": []}}],
            ),
            ToolMessage(
                content="<hide_in_user_ui> Plan made.",
                tool_call_id="tc1",
                name="write_plan",
            ),
        ]
    }


def _state_without_plan():
    """State where planner jumped straight to handoff without write_plan."""
    return {
        "messages": [
            AIMessage(content="I'll generate the images now."),
        ]
    }


def test_valid_plan_sets_plan_validated_true():
    result = validate_plan(_state_with_plan())
    assert result["plan_validated"] is True


def test_missing_plan_sets_plan_validated_false():
    result = validate_plan(_state_without_plan())
    assert result["plan_validated"] is False


def test_missing_plan_sets_error_context():
    result = validate_plan(_state_without_plan())
    assert result["error_context"]["type"] == "PLANNER_NO_PLAN"
