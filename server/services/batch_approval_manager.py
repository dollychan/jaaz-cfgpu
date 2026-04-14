# server/services/batch_approval_manager.py
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class BatchApprovalRequest:
    batch_id: str
    tool_calls: List[Dict[str, Any]]           # original tool calls shown to user
    approved_tool_calls: Optional[List[Dict[str, Any]]] = None  # set on approve
    _event: asyncio.Event = field(default_factory=asyncio.Event)


class BatchApprovalManager:
    """Manages async batch tool-call approval.

    Callers await request_approval(); the approval endpoint calls
    approve_batch() or reject_batch() to unblock them.
    Uses asyncio.Event (same pattern as ToolConfirmationManager).
    """

    DEFAULT_TIMEOUT = 300.0  # 5 minutes

    def __init__(self) -> None:
        self._pending: Dict[str, BatchApprovalRequest] = {}

    async def request_approval(
        self,
        batch_id: str,
        tool_calls: List[Dict[str, Any]],
        timeout: float = DEFAULT_TIMEOUT,
    ) -> List[Dict[str, Any]]:
        """Block until user approves/rejects or timeout. Returns approved calls (possibly modified)."""
        req = BatchApprovalRequest(batch_id=batch_id, tool_calls=tool_calls)
        self._pending[batch_id] = req
        try:
            await asyncio.wait_for(req._event.wait(), timeout=timeout)
            return req.approved_tool_calls if req.approved_tool_calls is not None else []
        except asyncio.TimeoutError:
            print(f"⏰ Batch approval timeout ({timeout}s), auto-rejecting: {batch_id}")
            return []
        finally:
            self._pending.pop(batch_id, None)

    def approve_batch(self, batch_id: str, approved_tool_calls: List[Dict[str, Any]]) -> bool:
        """Approve with optionally modified tool calls."""
        req = self._pending.get(batch_id)
        if req:
            req.approved_tool_calls = approved_tool_calls
            req._event.set()
            return True
        return False

    def reject_batch(self, batch_id: str) -> bool:
        """Reject all tool calls in the batch."""
        req = self._pending.get(batch_id)
        if req:
            req.approved_tool_calls = []
            req._event.set()
            return True
        return False

    def get_pending(self, batch_id: str) -> Optional[BatchApprovalRequest]:
        return self._pending.get(batch_id)


# Global singleton
batch_approval_manager = BatchApprovalManager()
