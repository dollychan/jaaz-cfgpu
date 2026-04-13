# type: ignore[import]
import traceback
from typing import Optional, List, Dict, Any, Callable, Awaitable
from langchain_core.messages import AIMessageChunk, ToolCall, convert_to_openai_messages, ToolMessage
from langgraph.graph import StateGraph
import json


class StreamProcessor:
    """流式处理器 - 负责处理智能体的流式输出"""

    def __init__(self, session_id: str, db_service: Any, websocket_service: Callable[[str, Dict[str, Any]], Awaitable[None]]):
        self.session_id = session_id
        self.db_service = db_service
        self.websocket_service = websocket_service
        self.tool_calls: List[ToolCall] = []
        self.last_saved_message_index = -1
        self.last_streaming_tool_call_id: Optional[str] = None
        # Set in process_stream; used by _handle_values_chunk for correct saving.
        # When context-window truncation reduces the input to K < M+1 messages,
        # new agent responses start at oai_messages[K], not at the DB count index M+1.
        self._langgraph_initial_count: int = 0
        self._new_responses_saved: int = 0
        # Number of stream chunks received before any exception.
        # Used by the retry logic in agent_service to decide if it is safe to
        # retry: chunks_received == 0 means the LLM failed before sending any
        # tokens, so the frontend state is unchanged and retry is safe.
        self.chunks_received: int = 0

    async def process_stream(self, swarm: StateGraph, messages: List[Dict[str, Any]], context: Dict[str, Any]) -> None:
        """处理整个流式响应

        Args:
            swarm: 智能体群组
            messages: 消息列表（可能已截断，仅用于发送给模型）
            context: 上下文信息
        """
        # 用 DB 中的实际条数确定起始索引，而非输入消息数量。
        # 输入消息可能因上下文截断而变短，若以其长度为基准会导致已有消息被重复写入。
        saved_messages = await self.db_service.get_chat_history(self.session_id)
        self.last_saved_message_index = len(saved_messages) - 1
        self._langgraph_initial_count = len(messages)
        self._new_responses_saved = 0
        print(f"📝 DB 已有 {len(saved_messages)} 条消息，LangGraph 输入 {len(messages)} 条，新消息从索引 {self.last_saved_message_index + 1} 开始")

        compiled_swarm = swarm.compile()

        try:
            async for chunk in compiled_swarm.astream(
                {"messages": messages},
                config=context,
                stream_mode=["messages", "custom", 'values']
            ):
                self.chunks_received += 1
                await self._handle_chunk(chunk)
        except Exception as e:
            err_str = str(e)
            # Handle API response errors (e.g., 'NoneType' object is not iterable)
            if isinstance(e, TypeError) and "'NoneType' object is not iterable" in err_str:
                print(f"❌ Upstream error detected: {e}")
                raise
            raise

    async def _handle_chunk(self, chunk: Any) -> None:
        # print('👇chunk', chunk)
        """处理单个chunk"""
        chunk_type = chunk[0]

        if chunk_type == 'values':
            await self._handle_values_chunk(chunk[1])
        else:
            await self._handle_message_chunk(chunk[1][0])

    async def _handle_values_chunk(self, chunk_data: Dict[str, Any]) -> None:
        """处理 values 类型的 chunk"""
        all_messages = chunk_data.get('messages', [])
        oai_messages = convert_to_openai_messages(all_messages)
        # 确保 oai_messages 是列表类型
        if not isinstance(oai_messages, list):
            oai_messages = [oai_messages] if oai_messages else []

        # 发送所有消息到前端
        await self.websocket_service(self.session_id, {
            'type': 'all_messages',
            'messages': oai_messages
        })

        # New agent responses start at oai_messages[_langgraph_initial_count].
        # Using DB count (last_saved_message_index + 1) as the start index fails
        # when context truncation makes K < M+1: the range becomes empty and
        # responses are never written to DB.
        new_responses = oai_messages[self._langgraph_initial_count:]
        for msg in new_responses[self._new_responses_saved:]:
            await self.db_service.create_message(
                self.session_id,
                msg.get('role', 'user'),
                json.dumps(msg)
            )
            self._new_responses_saved += 1

    async def _handle_message_chunk(self, ai_message_chunk: AIMessageChunk) -> None:
        """处理消息类型的 chunk"""
        # print('👇ai_message_chunk', ai_message_chunk)
        try:
            content = ai_message_chunk.content

            if isinstance(ai_message_chunk, ToolMessage):
                tool_content = ai_message_chunk.content if isinstance(ai_message_chunk.content, str) else ''

                # 工具调用结果之后会在 values 类型中发送到前端，这里会更快出现一些
                oai_message = convert_to_openai_messages([ai_message_chunk])[0]

                print('👇toolcall res oai_message', oai_message)
                await self.websocket_service(self.session_id, {
                    'type': 'tool_call_result',
                    'id': ai_message_chunk.tool_call_id,
                    'message': oai_message
                })
            elif content:
                # 发送文本内容
                await self.websocket_service(self.session_id, {
                    'type': 'delta',
                    'text': content
                })
            elif hasattr(ai_message_chunk, 'tool_calls') and ai_message_chunk.tool_calls and ai_message_chunk.tool_calls[0].get('name'):
                # 处理工具调用
                await self._handle_tool_calls(ai_message_chunk.tool_calls)

            # 处理工具调用参数流
            if hasattr(ai_message_chunk, 'tool_call_chunks'):
                await self._handle_tool_call_chunks(ai_message_chunk.tool_call_chunks)
        except Exception as e:
            print('🟠error', e)
            traceback.print_stack()

    async def _handle_tool_calls(self, tool_calls: List[ToolCall]) -> None:
        """处理工具调用"""
        self.tool_calls = [tc for tc in tool_calls if tc.get('name')]
        print('😘tool_call event', tool_calls)

        # 需要确认的工具列表
        TOOLS_REQUIRING_CONFIRMATION = {
            # 'generate_video_by_kling_v2_jaaz',
            # 'generate_video_by_seedance_v1_pro_volces',
            # 'generate_video_by_seedance_v1_lite_i2v',
            # 'generate_video_by_seedance_v1_lite_t2v',
            # 'generate_video_by_seedance_v1_jaaz',
            # 'generate_video_by_hailuo_02_jaaz',
            'generate_video_by_veo3_fast_jaaz',
        }

        for tool_call in self.tool_calls:
            tool_name = tool_call.get('name')

            # 检查是否需要确认
            if tool_name in TOOLS_REQUIRING_CONFIRMATION:
                # 对于需要确认的工具，不在这里发送事件，让工具函数自己处理
                print(
                    f'🔄 Tool {tool_name} requires confirmation, skipping StreamProcessor event')
                continue
            else:
                await self.websocket_service(self.session_id, {
                    'type': 'tool_call',
                    'id': tool_call.get('id'),
                    'name': tool_name,
                    'arguments': '{}'
                })

    async def _handle_tool_call_chunks(self, tool_call_chunks: List[Any]) -> None:
        """处理工具调用参数流"""
        for tool_call_chunk in tool_call_chunks:
            if tool_call_chunk.get('id'):
                # 标记新的流式工具调用参数开始
                self.last_streaming_tool_call_id = tool_call_chunk.get('id')
            else:
                if self.last_streaming_tool_call_id:
                    chunk_args = tool_call_chunk.get('args') or ''
                    await self.websocket_service(self.session_id, {
                        'type': 'tool_call_arguments',
                        'id': self.last_streaming_tool_call_id,
                        'text': chunk_args,
                    })
                else:
                    print('🟠no last_streaming_tool_call_id', tool_call_chunk)
