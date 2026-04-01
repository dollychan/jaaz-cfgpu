import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ToolConfirmationRequest:
    tool_call_id: str
    session_id: str
    tool_name: str
    arguments: Dict[str, Any]
    created_at: datetime
    confirmed: Optional[bool] = None
    _event: asyncio.Event = field(default_factory=asyncio.Event)


class ToolConfirmationManager:
    """工具确认管理器

    用 asyncio.Event 替代 busy-polling，避免每 0.1s 轮询带来的事件循环压力。
    request_confirmation 的 finally 块保证条目始终被清理，消除内存泄漏。
    超时后迟到的 confirm/cancel 调用会因字典中已无对应条目而安全返回 False。
    """

    TIMEOUT_SECONDS = 300  # 5 分钟

    def __init__(self):
        self.pending_confirmations: Dict[str, ToolConfirmationRequest] = {}

    async def request_confirmation(
        self,
        tool_call_id: str,
        session_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> bool:
        """请求工具确认，阻塞直到用户操作或超时（5 分钟），返回是否已确认。"""
        request = ToolConfirmationRequest(
            tool_call_id=tool_call_id,
            session_id=session_id,
            tool_name=tool_name,
            arguments=arguments,
            created_at=datetime.now(),
        )
        self.pending_confirmations[tool_call_id] = request

        try:
            await asyncio.wait_for(request._event.wait(), timeout=self.TIMEOUT_SECONDS)
            return request.confirmed is True
        except asyncio.TimeoutError:
            print(f"⏰ 工具确认超时 ({self.TIMEOUT_SECONDS}s)，自动取消: {tool_call_id}")
            return False
        finally:
            # 无论确认、取消还是超时，始终清理，防止内存泄漏
            self.pending_confirmations.pop(tool_call_id, None)

    def confirm_tool(self, tool_call_id: str) -> bool:
        """确认工具调用，立即唤醒等待协程。"""
        req = self.pending_confirmations.get(tool_call_id)
        if req:
            req.confirmed = True
            req._event.set()
            return True
        return False  # 超时后迟到的确认，字典已清理，安全忽略

    def cancel_confirmation(self, tool_call_id: str) -> bool:
        """取消工具调用，立即唤醒等待协程。"""
        req = self.pending_confirmations.get(tool_call_id)
        if req:
            req.confirmed = False
            req._event.set()
            return True
        return False

    def get_pending_request(self, tool_call_id: str) -> Optional[ToolConfirmationRequest]:
        """获取待确认的请求（用于调试或状态查询）。"""
        return self.pending_confirmations.get(tool_call_id)


# 全局实例
tool_confirmation_manager = ToolConfirmationManager()
