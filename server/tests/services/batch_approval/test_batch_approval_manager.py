import asyncio
import pytest
from services.batch_approval_manager import BatchApprovalManager


@pytest.mark.asyncio
async def test_approve_batch():
    mgr = BatchApprovalManager()
    batch_id = "test-batch-1"
    tool_calls = [{"id": "tc1", "name": "generate_image_by_flux", "args": {"prompt": "cat"}}]

    async def approve_after_delay():
        await asyncio.sleep(0.05)
        mgr.approve_batch(batch_id, tool_calls)

    asyncio.create_task(approve_after_delay())
    result = await mgr.request_approval(batch_id, tool_calls, timeout=2.0)
    assert result == tool_calls


@pytest.mark.asyncio
async def test_reject_batch():
    mgr = BatchApprovalManager()
    batch_id = "test-batch-2"
    tool_calls = [{"id": "tc2", "name": "generate_video_by_veo3_fast_jaaz", "args": {}}]

    async def reject_after_delay():
        await asyncio.sleep(0.05)
        mgr.reject_batch(batch_id)

    asyncio.create_task(reject_after_delay())
    result = await mgr.request_approval(batch_id, tool_calls, timeout=2.0)
    assert result == []


@pytest.mark.asyncio
async def test_modify_and_approve():
    mgr = BatchApprovalManager()
    batch_id = "test-batch-3"
    original = [{"id": "tc3", "name": "generate_video_by_veo3_fast_jaaz", "args": {"prompt": "original"}}]
    modified = [{"id": "tc3", "name": "generate_video_by_veo3_fast_jaaz", "args": {"prompt": "modified"}}]

    async def modify_after_delay():
        await asyncio.sleep(0.05)
        mgr.approve_batch(batch_id, modified)

    asyncio.create_task(modify_after_delay())
    result = await mgr.request_approval(batch_id, original, timeout=2.0)
    assert result[0]["args"]["prompt"] == "modified"


@pytest.mark.asyncio
async def test_timeout_returns_empty():
    mgr = BatchApprovalManager()
    result = await mgr.request_approval("timeout-batch", [{"id": "tc4", "name": "x", "args": {}}], timeout=0.1)
    assert result == []
