from models.tool_model import ToolInfoJson
from services.db_service import db_service
from .StreamProcessor import StreamProcessor
from .configs import PlannerAgentConfig, create_handoff_tool
import asyncio
import traceback
import json
import os
from utils.http_client import HttpClient
from langgraph_swarm import create_swarm  # type: ignore
from langgraph.prebuilt import create_react_agent, ToolNode  # type: ignore
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools import tool as lc_tool  # type: ignore
from langchain_core.callbacks import BaseCallbackHandler
from services.websocket_service import send_to_websocket  # type: ignore
from services.config_service import config_service
from services.settings_service import settings_service
from services.tool_service import tool_service
from typing import Optional, List, Dict, Any, cast, Set, TypedDict
from models.config_model import ModelInfo


class ContextInfo(TypedDict):
    """Context information passed to tools"""
    canvas_id: str
    session_id: str
    model_info: Dict[str, List[ModelInfo]]


class PromptLoggingCallbackHandler(BaseCallbackHandler):
    """Callback handler that logs the full prompt sent to the LLM on EVERY invocation.
    
    This captures:
    - The initial prompt (system + messages)
    - Every re-invocation after tool calls (system + messages + tool results)
    """
    
    def __init__(self):
        self.call_count = 0
    
    def _log_messages(self, messages: List[Any], prefix: str = "") -> None:
        """格式化打印消息列表"""
        for i, msg in enumerate(messages):
            role = getattr(msg, 'type', getattr(msg, '__class__.__name__', 'unknown'))
            name = getattr(msg, 'name', '')
            content = getattr(msg, 'content', '')
            tool_calls = getattr(msg, 'tool_calls', None)
            tool_call_id = getattr(msg, 'tool_call_id', None)
            
            header = f"[{i}] role={role}"
            if name:
                header += f", name={name}"
            if tool_call_id:
                header += f", tool_call_id={tool_call_id}"
            if tool_calls:
                tc_names = [tc.get('name', '?') for tc in tool_calls] if isinstance(tool_calls, list) else []
                header += f", tool_calls=[{', '.join(tc_names)}]"
            
            print(f"\n  {header}")
            
            # 打印内容
            if isinstance(content, str):
                preview = content if len(content) <= 2000 else content[:2000] + f"... [截断, 总{len(content)}字符]"
                print(f"    content: {preview}")
            elif isinstance(content, list):
                for j, item in enumerate(content):
                    if isinstance(item, dict):
                        if item.get('type') == 'text':
                            text = item.get('text', '')
                            preview = text if len(text) <= 1000 else text[:1000] + f"... [截断, 总{len(text)}字符]"
                            print(f"    content[{j}] (text): {preview}")
                        elif item.get('type') == 'image_url':
                            url = item.get('image_url', {}).get('url', '')[:120]
                            print(f"    content[{j}] (image_url): {url}")
                        else:
                            print(f"    content[{j}] ({item.get('type')}): {str(item)[:200]}")
                    elif hasattr(item, 'content'):
                        text = item.content if isinstance(item.content, str) else str(item.content)
                        preview = text if len(text) <= 1000 else text[:1000] + f"... [截断]"
                        print(f"    content[{j}] ({getattr(item, 'type', 'unknown')}): {preview}")
                    else:
                        print(f"    content[{j}]: {str(item)[:200]}")
            
            # 打印 tool_calls
            if tool_calls and isinstance(tool_calls, list):
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        tc_name = tc.get('name', '?')
                        tc_args = tc.get('args', {})
                        tc_id = tc.get('id', '?')
                    else:
                        tc_name = getattr(tc, 'name', '?')
                        tc_args = getattr(tc, 'args', {})
                        tc_id = getattr(tc, 'id', '?')
                    tc_args_str = json.dumps(tc_args, ensure_ascii=False) if tc_args else '{}'
                    if len(tc_args_str) > 1000:
                        tc_args_str = tc_args_str[:1000] + '... [截断]'
                    print(f"    tool_call: {tc_id} -> {tc_name}({tc_args_str})")

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[Any], **kwargs: Any) -> None:
        """Called when the chat model starts. Logs the full prompt on EVERY invocation."""
        self.call_count += 1
        print(f"\n{'='*100}")
        print(f"📝 [LLM CALLBACK LOG] Call #{self.call_count}")
        print(f"{'='*100}")
        
        # messages 通常是 List[List[BaseMessage]] 格式
        for msg_group_idx, msg_group in enumerate(messages):
            if isinstance(msg_group, list):
                print(f"\n{'─'*60}")
                print(f"  MESSAGE GROUP {msg_group_idx} ({len(msg_group)} 条消息):")
                print(f"{'─'*60}")
                self._log_messages(msg_group)
        
        print(f"\n{'='*100}\n")

    run_on_chat_model_start = on_chat_model_start

    async def on_chat_model_start_async(self, serialized: Dict[str, Any], messages: List[Any], **kwargs: Any) -> None:
        """Async version of on_chat_model_start."""
        self.on_chat_model_start(serialized, messages, **kwargs)


def _log_llm_prompt(messages: List[Dict[str, Any]], system_prompt: str, agent_name: str = "assistant") -> None:
    """打印发给 LLM 的完整初始 prompt（包含 system prompt + 消息历史），用于分析。
    
    Args:
        messages: 消息历史（已修复/截断/展开的图片URL）
        system_prompt: 系统提示词
        agent_name: 当前 agent 名称
    """
    print(f"\n{'='*100}")
    print(f"📝 [LLM PROMPT LOG] Agent: {agent_name}")
    print(f"{'='*100}")
    
    # 打印 System Prompt
    print(f"\n{'─'*60}")
    print(f"  SYSTEM PROMPT (前 3000 字符):")
    print(f"{'─'*60}")
    if system_prompt:
        if len(system_prompt) > 3000:
            print(system_prompt[:3000])
            print(f"\n... [已截断，总长度 {len(system_prompt)} 字符]")
        else:
            print(system_prompt)
    else:
        print("  (无 system prompt)")
    
    # 打印消息历史
    print(f"\n{'─'*60}")
    print(f"  MESSAGE HISTORY ({len(messages)} 条消息):")
    print(f"{'─'*60}")
    
    for i, msg in enumerate(messages):
        role = msg.get('role', 'unknown')
        name = msg.get('name', '')
        content = msg.get('content', '')
        tool_calls = msg.get('tool_calls')
        tool_call_id = msg.get('tool_call_id')
        
        header = f"[{i}] role={role}"
        if name:
            header += f", name={name}"
        if tool_call_id:
            header += f", tool_call_id={tool_call_id}"
        if tool_calls:
            header += f", tool_calls=[{', '.join(tc.get('name', '?') for tc in tool_calls)}]"
        
        print(f"\n  {header}")
        
        # 打印内容
        if isinstance(content, str):
            preview = content if len(content) <= 2000 else content[:2000] + f"... [截断, 总{len(content)}字符]"
            print(f"    content: {preview}")
        elif isinstance(content, list):
            for j, item in enumerate(content):
                if item.get('type') == 'text':
                    text = item.get('text', '')
                    preview = text if len(text) <= 1000 else text[:1000] + f"... [截断, 总{len(text)}字符]"
                    print(f"    content[{j}] (text): {preview}")
                elif item.get('type') == 'image_url':
                    url = item.get('image_url', {}).get('url', '')[:120]
                    print(f"    content[{j}] (image_url): {url}")
                else:
                    print(f"    content[{j}] ({item.get('type')}): {str(item)[:200]}")
        
        # 打印 tool_calls
        if tool_calls:
            for tc in tool_calls:
                tc_name = tc.get('name', '?')
                tc_args = tc.get('args', {})
                tc_args_str = json.dumps(tc_args, ensure_ascii=False)
                if len(tc_args_str) > 1000:
                    tc_args_str = tc_args_str[:1000] + '... [截断]'
                print(f"    tool_call: {tc.get('id')} -> {tc_name}({tc_args_str})")
    
    print(f"\n{'='*100}\n")


def _fix_chat_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """修复聊天历史中不完整的工具调用

    根据LangGraph文档建议，移除没有对应ToolMessage的tool_calls。
    参考: https://langchain-ai.github.io/langgraph/troubleshooting/errors/INVALID_CHAT_HISTORY/

    注意：当 assistant 消息同时有 content 和 valid_tool_calls 时，保留两者。
    只移除 tool_calls 而保留对应的 ToolMessage 会制造孤儿 ToolMessage（违反消息
    格式规范），反而让 LLM 无法建立调用-结果因果链。LangGraph 不会重新执行历史
    tool_calls，因此原样保留是安全且正确的。
    """
    if not messages:
        return messages

    fixed_messages: List[Dict[str, Any]] = []
    tool_call_ids: Set[str] = set()

    # 第一遍：收集所有ToolMessage的tool_call_id
    for msg in messages:
        if msg.get('role') == 'tool' and msg.get('tool_call_id'):
            tool_call_id = msg.get('tool_call_id')
            if tool_call_id:
                tool_call_ids.add(tool_call_id)

    # 第二遍：修复AIMessage中的tool_calls
    for msg in messages:
        if msg.get('role') == 'assistant' and msg.get('tool_calls'):
            valid_tool_calls: List[Dict[str, Any]] = []
            removed_calls: List[str] = []

            for tool_call in msg.get('tool_calls', []):
                tool_call_id = tool_call.get('id')
                if tool_call_id not in tool_call_ids:
                    if tool_call_id:
                        removed_calls.append(tool_call_id)
                    continue

                # Skip phantom tool_calls with empty or missing type/function fields.
                # The planner LLM sometimes emits tool_calls with type='' or no function.
                tc_type = tool_call.get('type', '')
                fn = tool_call.get('function')
                if not tc_type:
                    tc_type = 'function'
                    tool_call['type'] = 'function'
                if not fn or not fn.get('name'):
                    removed_calls.append(tool_call_id or '(no_id)')
                    continue

                # Qwen/Dashscope requires function.arguments to be a non-null string.
                # Fix any tool_calls with missing or null arguments.
                if fn is None:
                    fn = {}
                    tool_call['function'] = fn
                if fn.get('arguments') is None:
                    fn['arguments'] = '{}'

                valid_tool_calls.append(tool_call)

            has_content = bool(msg.get('content'))

            if removed_calls:
                print(
                    f"🔧 修复消息历史：移除了 {len(removed_calls)} 个不完整的工具调用: {removed_calls}")

            if valid_tool_calls:
                # Keep valid tool_calls intact together with any content.
                # Removing tool_calls while keeping the corresponding ToolMessages
                # creates orphan ToolMessages (invalid format) and makes the LLM
                # unable to correlate results with calls — the opposite of what we want.
                # LangGraph does NOT re-execute historical tool_calls on the next turn,
                # so leaving them in place is safe and provides correct context.
                msg_copy = msg.copy()
                msg_copy['tool_calls'] = valid_tool_calls
                fixed_messages.append(msg_copy)
            elif has_content:
                msg_copy = msg.copy()
                msg_copy.pop('tool_calls', None)
                fixed_messages.append(msg_copy)
        else:
            fixed_messages.append(msg)

    return fixed_messages


def _tc_name(tc: dict) -> str:
    """从 LangChain 或 OpenAI 格式的 tool_call dict 中提取工具名。

    LangChain 内部格式: {'name': '...', 'args': {...}, 'id': '...', 'type': 'function'}
    OpenAI API 格式:    {'function': {'name': '...', 'arguments': '...'}, 'id': '...', 'type': 'function'}
    """
    return tc.get('name') or (tc.get('function') or {}).get('name') or ''


def _fix_tc_type(tc: dict) -> None:
    """确保 tool_call 的 type 字段为 'function'（OpenAI 格式要求）。

    LangChain 内部可能用 'tool_call'，但 OpenAI API 的 Pydantic 模型
    要求 type 必须是 'function'，否则序列化会报 PydanticSerializationUnexpectedValue。
    """
    if not tc.get('type') or tc.get('type') != 'function':
        tc['type'] = 'function'


def _pre_model_hook(state: dict) -> dict:
    """在每次 LLM 调用前修复 state 中历史 AIMessage 的 tool_calls。

    LangGraph 在 streaming 多轮对话中会累积新的 assistant 消息，
    这些消息可能包含 LLM 产生的空 name 或空 args 的 tool_calls。
    此 hook 在每轮调用前清理它们，避免 Qwen API 报错。

    注意：只修复格式问题（空 args、错误的 type），不删除 tool_calls。
    删除 tool_calls 会导致 LangGraph 状态损坏。

    同时兼容 LangChain 内部格式（name/args）和 OpenAI 格式（function.name）。
    """
    from langchain_core.messages import AIMessage
    msgs = state.get('messages', [])
    for msg in msgs:
        if not isinstance(msg, AIMessage):
            continue
        if not getattr(msg, 'tool_calls', None):
            continue
        for tc in msg.tool_calls:
            # 修复格式问题，但不删除 tool_call
            _fix_tc_type(tc)
            # LangChain 格式：确保 args 不为 None 或空字符串
            if 'args' in tc:
                if tc['args'] is None or tc['args'] == '':
                    tc['args'] = {}
            # OpenAI 格式：确保 arguments 不为 None 或空字符串
            fn = tc.get('function')
            if fn is not None:
                if fn.get('arguments') is None or fn.get('arguments') == '':
                    fn['arguments'] = '{}'
    return state


def _make_safe_tool_node(tools: list, **kwargs) -> tuple:
    """创建安全的 ToolNode，自动跳过无效的 tool_call（空名称、不存在的工具）。

    Returns:
        tuple: (ToolNode, post_model_hook | None)
        - 如果 langgraph-prebuilt 支持 awrap_tool_call，返回 (ToolNode, None)
        - 否则返回 (ToolNode, None) - 旧版本不使用 post_model_hook，避免状态损坏

    注意：旧版本 langgraph-prebuilt 没有 awrap_tool_call 参数。
    直接修改 AIMessage.tool_calls 会导致 "No AIMessage found in input" 错误。
    因此旧版本只能依赖 ToolNode 的默认错误处理，让 planner prompt 中的 ERROR HANDLING
    指导 LLM 在收到错误后如何恢复。
    """
    from langchain_core.tools import BaseTool
    from langchain_core.messages import ToolMessage
    import inspect

    # 收集所有已注册的工具名称
    valid_tool_names: set[str] = set()
    for t in tools:
        if isinstance(t, BaseTool):
            valid_tool_names.add(t.name)
        elif hasattr(t, 'name'):
            valid_tool_names.add(t.name)

    async def _awrap_tool_call(request, execute):
        """awrap_tool_call wrapper that receives ToolCallRequest (dataclass).

        Args:
            request: ToolCallRequest with fields: tool_call (dict), tool (BaseTool|None), state, runtime
            execute: Async callable to execute the tool
        """
        tool_call = request.tool_call
        name = _tc_name(tool_call)
        tool_call_id = tool_call.get('id', 'unknown')

        # 跳过空名称的工具调用
        if not name:
            return ToolMessage(
                content='<skipped> Empty tool_call ignored',
                tool_call_id=tool_call_id,
                name='skipped',
            )
        # 跳过未注册的工具调用（如幽灵空 tool_call 残留的名称）
        if name not in valid_tool_names:
            return ToolMessage(
                content=f'<skipped> Tool "{name}" is not registered, ignored',
                tool_call_id=tool_call_id,
                name='skipped',
            )
        return await execute(request)

    # 检查 ToolNode 是否支持 awrap_tool_call 参数（langgraph-prebuilt 1.0+）
    tool_node_params = inspect.signature(ToolNode.__init__).parameters
    if 'awrap_tool_call' in tool_node_params:
        # 新版本：使用 awrap_tool_call 拦截无效 tool_calls
        return ToolNode(tools, awrap_tool_call=_awrap_tool_call, **kwargs), None
    else:
        # 旧版本：不使用 post_model_hook（会导致状态损坏）
        # 让 ToolNode 的默认错误处理机制处理无效 tool_calls
        # planner prompt 中的 ERROR HANDLING 部分会指导 LLM 如何恢复
        return ToolNode(tools, **kwargs), None


def _post_model_hook(state: dict, tools: list) -> dict:
    """【已弃用】不再使用 post_model_hook 过滤 tool_calls。

    直接修改 msg.tool_calls 会破坏 LangGraph 内部状态对象，
    导致 ToolNode._parse_input 找不到 AIMessage（No AIMessage found in input）。
    改为在 ToolNode 层用 awrap_tool_call 处理无效调用。
    """
    return state


def _parse_input_media_from_messages(messages: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """从最后一条用户消息中解析所有媒体引用，返回按顺序排列的有序列表。

    支持以下格式（按前端实际发送的格式）：
    1. <input_images><image file_id="im_xxx"/></input_images> XML 标签
    2. <input_videos><video file_id="vi_xxx"/></input_videos> XML 标签
    3. <input_audios><audio file_id="au_xxx"/></input_audios> XML 标签
    4. asset:// URL 嵌入的 JSON 结构（image_url / video_url / audio_url type）
    5. 裸文件 ID（im_xxxx.png / vi_xxxx.mp4）出现在 prompt 文本中

    解析顺序即前端插入顺序，不做任何重排。
    """
    import re

    # 找最后一条 user 消息
    last_user = next(
        (m for m in reversed(messages) if m.get('role') == 'user'),
        None,
    )
    if not last_user:
        return {}

    # 提取纯文本
    content = last_user.get('content', '')
    if isinstance(content, list):
        text = ' '.join(
            item.get('text', '') for item in content
            if isinstance(item, dict) and item.get('type') == 'text'
        )
    else:
        text = str(content)

    result: Dict[str, List[str]] = {}

    # 1. XML 标签格式（file_id 属性，保留出现顺序）
    for media_type, xml_tag in [
        ('input_images', 'image'),
        ('input_videos', 'video'),
        ('input_audios', 'audio'),
    ]:
        ids = re.findall(
            rf'<{xml_tag}[^>]+file_id=["\']([^"\']+)["\']',
            text,
        )
        if ids:
            result[media_type] = ids

    # 2. asset:// URL JSON 结构（type 字段决定桶）
    asset_type_map = {'image_url': 'input_images', 'video_url': 'input_videos', 'audio_url': 'input_audios'}
    for json_type, bucket in asset_type_map.items():
        assets = re.findall(
            rf'"type":\s*"{re.escape(json_type)}"[^{{}}]*?"url":\s*"(asset://[^"]+)"',
            text,
        )
        if assets:
            result.setdefault(bucket, []).extend(assets)

    # 3. 裸文件 ID（im_ / vi_ / au_ 前缀，补充 XML 未覆盖的情况）
    if 'input_images' not in result:
        bare_images = re.findall(r'\bim_[A-Za-z0-9_.-]+', text)
        if bare_images:
            result['input_images'] = bare_images
    if 'input_videos' not in result:
        bare_videos = re.findall(r'\bvi_[A-Za-z0-9_.-]+', text)
        if bare_videos:
            result['input_videos'] = bare_videos

    return result


def _format_media_manifest(media: Dict[str, List[str]]) -> str:
    """将预解析的媒体引用格式化为注入 LLM prompt 的清单文本。"""
    if not media:
        return ''
    lines = ['INPUT MEDIA MANIFEST (server-parsed — authoritative order):']
    for bucket, ids in media.items():
        for i, fid in enumerate(ids):
            lines.append(f'  {bucket}[{i}]: {fid}')
        lines.append(f'  → full list: {ids}')
    lines += [
        'CRITICAL RULES:',
        '  1. Pass the FULL list above as the corresponding parameter (input_images / input_videos / input_audios).',
        '  2. Do NOT reorder, drop, or add extra IDs.',
        '  3. If a step only needs a subset, slice by index (e.g. input_images[0]) but preserve relative order.',
    ]
    return '\n'.join(lines)


def _expand_image_urls(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将消息中的相对图片 URL 展开为可供外部模型 API 访问的绝对 URL。

    前端把图片存为相对路径（/api/file/{id} 或 /api/material/serve/{name}）
    而非 base64。此函数在消息发给模型前将其补全为绝对 URL，令远端 API 可以访问。
    已是 http/https 的 URL 直接透传。
    """
    # video_generation_core._get_server_base_url() 是同一逻辑，待后续统一到共享工具模块
    server_base = os.environ.get("JAAZ_SERVER_URL", "http://127.0.0.1:57988").rstrip("/")
    result: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.get('content')
        if isinstance(content, list):
            new_content = []
            for item in content:
                if item.get('type') == 'image_url':
                    url = item.get('image_url', {}).get('url', '')
                    if url.startswith('/api/'):
                        item = {**item, 'image_url': {**item['image_url'], 'url': f"{server_base}{url}"}}
                new_content.append(item)
            msg = {**msg, 'content': new_content}
        result.append(msg)
    return result


def _log_payload_size(messages: List[Dict[str, Any]], approx_tokens: Optional[int] = None) -> None:
    """统计并打印发送给模型的消息历史大小，便于监测上下文长度。

    approx_tokens: 如果调用方已经计算过 token 估算值，直接传入避免重复序列化。
    """
    if approx_tokens is None:
        byte_size = len(json.dumps(messages, ensure_ascii=False).encode('utf-8'))
        approx_tokens = byte_size // 4
    else:
        byte_size = approx_tokens * 4
    print(
        f"📊 模型请求 payload: {len(messages)} 条消息  "
        f"≈{approx_tokens:,} tokens  ({byte_size / 1024:.1f} KB)"
    )


DEFAULT_CONTEXT_WINDOW = 128000  # Default context window size for text models
MAX_SINGLE_MESSAGE_TOKENS = 32000  # Maximum tokens for a single message


def _estimate_message_tokens(message: Dict[str, Any]) -> int:
    """估算单条消息的token数量"""
    return len(json.dumps(message, ensure_ascii=False).encode('utf-8')) // 4


def _truncate_single_message(
    message: Dict[str, Any],
    max_tokens: int = MAX_SINGLE_MESSAGE_TOKENS
) -> Dict[str, Any]:
    """截断超大的单条消息（主要针对超长文本内容）。

    图片以 URL 形式存储，不会引发单条消息过大；过大通常来自 tool 返回的超长文本。
    """
    msg_tokens = _estimate_message_tokens(message)
    if msg_tokens <= max_tokens:
        return message

    print(f"⚠️ 单条消息过大 ({msg_tokens:,} tokens > {max_tokens:,})，正在截断...")

    truncated_msg = message.copy()
    content = truncated_msg.get('content')

    if isinstance(content, list):
        new_content = []
        for item in content:
            if item.get('type') == 'text':
                text = item.get('text', '')
                if len(text.encode('utf-8')) // 4 > max_tokens // 2:
                    new_content.append({
                        'type': 'text',
                        'text': text[:max_tokens * 2] + '...[文本已截断]'
                    })
                else:
                    new_content.append(item)
            else:
                new_content.append(item)
        truncated_msg['content'] = new_content
    elif isinstance(content, str):
        if len(content.encode('utf-8')) // 4 > max_tokens:
            truncated_msg['content'] = content[:max_tokens * 4] + '...[内容已截断]'

    final_tokens = _estimate_message_tokens(truncated_msg)
    print(f"✅ 单条消息截断完成：{msg_tokens:,} → {final_tokens:,} tokens")
    return truncated_msg


def _truncate_messages_by_context_window(
    messages: List[Dict[str, Any]],
    context_window: int = DEFAULT_CONTEXT_WINDOW
) -> tuple:
    """根据上下文窗口大小截断消息历史，返回 (messages, approx_tokens)。

    保留第一条消息（system prompt）和尽可能多的最近消息。
    超大的单条消息先做内容截断，再做历史窗口裁剪。
    """
    if not messages:
        return messages, 0

    truncated_messages = [_truncate_single_message(msg, MAX_SINGLE_MESSAGE_TOKENS) for msg in messages]

    approx_tokens = len(json.dumps(truncated_messages, ensure_ascii=False).encode('utf-8')) // 4
    if approx_tokens <= context_window:
        return truncated_messages, approx_tokens

    print(f"⚠️ 上下文超限 ({approx_tokens:,} tokens > {context_window:,})，正在截断消息历史...")

    first_msg = truncated_messages[0]
    remaining_msgs = truncated_messages[1:]

    # 从最新消息向前贪心填充，直到预算耗尽。
    # 用 continue（而非 break）跳过超大的单条消息，避免一条大消息截断所有更早的内容。
    budget = context_window - (len(json.dumps(first_msg, ensure_ascii=False).encode('utf-8')) // 4)
    kept: List[Dict[str, Any]] = []
    for msg in reversed(remaining_msgs):
        msg_tokens = len(json.dumps(msg, ensure_ascii=False).encode('utf-8')) // 4
        if msg_tokens <= budget:
            kept.append(msg)
            budget -= msg_tokens

    result = [first_msg] + kept[::-1]
    final_tokens = len(json.dumps(result, ensure_ascii=False).encode('utf-8')) // 4
    print(f"✅ 截断完成：保留 {len(result)}/{len(truncated_messages)} 条消息 (≈{final_tokens:,} tokens)")
    return result, final_tokens


def _create_text_model(text_model: ModelInfo) -> Any:
    """创建语言模型实例"""
    model = text_model.get('model')
    provider = text_model.get('provider')
    url = text_model.get('url')
    # api_key 优先级：model_info 显式传入 > config_service provider 配置
    api_key = text_model.get('api_key') or config_service.app_config.get(  # type: ignore
        provider, {}).get("api_key", "")

    if provider == 'ollama':
        return ChatOllama(
            model=model,
            base_url=url,
        )
    else:
        # 验证 API key 是否存在
        if not api_key or not api_key.strip():
            raise ValueError(
                f"API key is missing for provider '{provider}'. "
                f"Please configure the API key in config.toml under [{provider}].api_key, "
                f"or set it in settings.json builtin_model.api_key"
            )
        
        http_client = HttpClient.create_sync_client()
        http_async_client = HttpClient.create_async_client()
        # cfgpu streaming mode ignores tools and returns plain text instead of
        # tool_calls, causing generate_from_stream to get 0 valid chunks.
        # disable_streaming="tool_calling" uses non-streaming _agenerate only
        # when tools are bound, preserving token-by-token streaming for text.
        disable_streaming = "tool_calling" if provider == 'cfgpu' else False
        return ChatOpenAI(
            model=model,
            api_key=api_key,  # type: ignore
            timeout=300,
            base_url=url,
            temperature=0,
            disable_streaming=disable_streaming,
            http_client=http_client,
            http_async_client=http_async_client
        )


def _create_builtin_model() -> Optional[Any]:
    """从 settings + config.toml 读取并创建内置编排模型实例

    优先级：
    1. settings.json builtin_model.provider / model / api_key / url
    2. model / api_key / url 为空时，从 config.toml 对应 provider 的配置中补全
       - model 为空：取 config.toml 该 provider 下第一个 type='text' 的 model
       - api_key 为空：取 config.toml 该 provider 的 api_key
       - url 为空：取 config.toml 该 provider 的 url
    3. 仍无法解析 provider/model 则返回 None，调用方 fallback 到用户选择的 text tool
    """
    raw_settings = settings_service.get_raw_settings()
    builtin_config = raw_settings.get('builtin_model', {})

    provider = builtin_config.get('provider', '').strip()
    model = builtin_config.get('model', '').strip()
    url = builtin_config.get('url', '').strip()
    api_key = builtin_config.get('api_key', '').strip()

    if not provider:
        print("⚠️ builtin_model 未配置 provider，将 fallback 到用户选择的 text model tool 作为编排器")
        return None

    # 从 config.toml 对应 provider 配置中补全缺失字段
    provider_config = config_service.app_config.get(provider, {})
    if not api_key:
        api_key = provider_config.get('api_key', '').strip()
    if not url:
        url = provider_config.get('url', '').strip()
    if not model:
        # 取该 provider 下第一个 type='text' 的 model
        for model_name, model_cfg in provider_config.get('models', {}).items():
            if model_cfg.get('type') == 'text':
                model = model_name
                break

    if not model:
        print(f"⚠️ builtin_model provider={provider} 下未找到 text model，将 fallback 到用户选择的 text model tool")
        return None

    print(f"🤖 使用内置编排模型: {provider}/{model}")

    model_info: ModelInfo = {
        'provider': provider,
        'model': model,
        'url': url,
        'type': 'text',
        'api_key': api_key,  # 传入已解析的 key，避免 _create_text_model 重读 config 时丢失
    }
    return _create_text_model(model_info)


def _create_text_generation_tool(model_info: Dict[str, Any]) -> Optional[Any]:
    """把一个 text model 包装成 LangChain tool

    Args:
        model_info: 包含 provider / model (或 id) / url / display_name 的字典

    Returns:
        LangChain BaseTool 或 None（创建失败时）
    """
    provider = model_info.get('provider', '').strip()
    # text 类型工具的 id 就是 model name；也接受 'model' 字段（来自 ModelInfo）
    model_name = (model_info.get('model') or model_info.get('id') or '').strip()
    display_name = (model_info.get('display_name') or model_name).strip()

    if not provider or not model_name:
        print(f"⚠️ 无法创建 text tool：provider 或 model 为空 {model_info}")
        return None

    # URL fallback
    url = (model_info.get('url') or '').strip()
    if not url:
        url = config_service.app_config.get(provider, {}).get('url', '')

    lm: ModelInfo = {'provider': provider, 'model': model_name, 'url': url, 'type': 'text'}
    try:
        text_llm = _create_text_model(lm)
    except Exception as e:
        print(f"⚠️ 创建 text model 失败 {provider}/{model_name}: {e}")
        return None

    # 工具名需是合法 Python 标识符
    safe = (
        model_name
        .replace('/', '_').replace('-', '_').replace('.', '_')
        .replace(':', '_').replace(' ', '_')
    )
    tool_name = f"generate_text_with_{provider}_{safe}"

    @lc_tool(
        tool_name,
        description=(
            f"Use {display_name} for complex text generation, deep reasoning, "
            "creative writing, detailed analysis, summarization, and "
            "knowledge-intensive tasks. Call this when high-quality language "
            "output is needed beyond simple prompt/parameter extraction."
        )
    )
    async def text_gen_tool(prompt: str) -> str:
        resp = await text_llm.ainvoke([HumanMessage(content=prompt)])
        return str(resp.content)

    return text_gen_tool


async def _collect_tools(
    tool_list: List[ToolInfoJson],
    text_model: ModelInfo,
    has_text_model: bool,
) -> tuple:
    """构建并按类型分流工具：text → Planner；image/video → Creator。

    Returns:
        (text_lc_tools, media_lc_tools)
    """
    text_lc_tools: List[Any] = []
    media_lc_tools: List[Any] = []

    if has_text_model:
        text_tool = _create_text_generation_tool(dict(text_model))
        if text_tool:
            text_lc_tools.append(text_tool)
            print(f"✅ 添加 text tool（来自 text_model）: {text_tool.name}")

    for tool_json in (tool_list or []):
        tool_type = tool_json.get('type', '')
        tool_id = tool_json.get('id', '')

        if tool_type == 'text':
            txt_tool = _create_text_generation_tool(dict(tool_json))
            if txt_tool:
                text_lc_tools.append(txt_tool)
                print(f"✅ 添加 text tool（来自 tool_list）: {txt_tool.name}")
        else:
            lc_t = tool_service.get_tool(tool_id)
            if lc_t:
                media_lc_tools.append(lc_t)
            else:
                print(f"⚠️ 工具未找到: {tool_id}")

    print(f"📝 Text tools (→ Planner): {[t.name for t in text_lc_tools]}")
    print(f"🎨 Media tools (→ Creator): {[t.name for t in media_lc_tools]}")
    return text_lc_tools, media_lc_tools


async def _ensure_orchestrator(
    orchestrator: Any,
    text_lc_tools: List[Any],
    text_model: ModelInfo,
    has_text_model: bool,
    tool_list: List[ToolInfoJson],
) -> tuple:
    """若内置模型未配置，fallback 到第一个 text tool 作为编排器。

    Returns:
        (orchestrator, text_lc_tools) — orchestrator 可能从 None 变为实例，
        text_lc_tools 可能移除了作为编排器的那个 tool。
    """
    if orchestrator:
        return orchestrator, text_lc_tools

    if not text_lc_tools:
        return None, text_lc_tools

    if has_text_model:
        orchestrator = _create_text_model(text_model)
        provider = text_model.get('provider', '')
        model_id = text_model.get('model') or text_model.get('id', '')
        safe_id = model_id.replace('/', '_').replace('-', '_').replace('.', '_').replace(':', '_').replace(' ', '_')
        text_model_tool_name = f"generate_text_with_{provider}_{safe_id}"
        text_lc_tools = [t for t in text_lc_tools if t.name != text_model_tool_name]
        print(f"⚠️ builtin_model 未配置，使用 text_model 作为编排器（向后兼容模式）")
    else:
        first_text_tool_json = next(
            (t for t in (tool_list or []) if t.get('type') == 'text'), None
        )
        if first_text_tool_json:
            provider = first_text_tool_json.get('provider', '')
            model_name = first_text_tool_json.get('id', '')
            url = config_service.app_config.get(provider, {}).get('url', '')
            lm: ModelInfo = {'provider': provider, 'model': model_name, 'url': url, 'type': 'text'}
            orchestrator = _create_text_model(lm)
            text_lc_tools = text_lc_tools[1:]
            print(f"⚠️ builtin_model 未配置，使用 {model_name} 作为编排器（fallback 模式）")

    return orchestrator, text_lc_tools


def _build_creator_prompt(
    tool_list: List[ToolInfoJson],
    system_prompt: Optional[str],
) -> str:
    """构建 Creator agent 的 system prompt。

    自定义 prompt 会追加安全附录；否则使用完整默认 prompt。
    """
    from .configs.image_video_creator_config import ImageVideoCreatorAgentConfig
    creator_config = ImageVideoCreatorAgentConfig(tool_list or [])

    if system_prompt:
        return system_prompt + "\n\n" + creator_config.custom_prompt_appendix
    return creator_config.system_prompt


def _build_planner_agent(
    text_tools_in_list: List[ToolInfoJson],
    text_lc_tools: List[Any],
) -> tuple:
    """创建 Planner agent 及其 prompt。

    Returns:
        (planner_agent, planner_prompt)
    """
    print(f"\n{'='*100}")
    print(f"🏗️  构建 Planner Agent")
    print(f"{'─'*100}")

    first_text_json = text_tools_in_list[0]
    planner_model_info: ModelInfo = {
        'provider': first_text_json.get('provider', ''),
        'model': first_text_json.get('id', ''),
        'url': config_service.app_config.get(
            first_text_json.get('provider', ''), {}
        ).get('url', ''),
        'type': 'text',
    }
    planner_lm = _create_text_model(planner_model_info)

    print(f"\n{'='*80}")
    print(f"🔧 获取 Planner 工具:")
    print(f"{'─'*80}")

    # 获取 write_plan 工具
    write_plan_lc = tool_service.get_tool('write_plan')
    if write_plan_lc:
        print(f"  ✅ write_plan 工具获取成功: {write_plan_lc.name if hasattr(write_plan_lc, 'name') else 'unnamed'}")
    else:
        print(f"  ❌ write_plan 工具获取失败 (返回 None)")

    # 创建 handoff 工具
    handoff_to_creator = create_handoff_tool(
        agent_name='assistant',
        description='Transfer to the image/video creator agent to execute the plan.',
    )
    if handoff_to_creator:
        print(f"  ✅ handoff 工具创建成功: {handoff_to_creator.name}")
    else:
        print(f"  ❌ handoff 工具创建失败 (返回 None)")

    # 收集 text 工具
    print(f"  📋 text_lc_tools 数量: {len(text_lc_tools)}")
    if text_lc_tools:
        print(f"  📦 text_lc_tools: {[t.name if hasattr(t, 'name') else 'unnamed' for t in text_lc_tools]}")

    # 更新 prompt 中的工具名称以匹配实际工具
    handoff_tool_name = handoff_to_creator.name  # 实际工具名称: transfer_to_assistant
    planner_tools = [t for t in [write_plan_lc, handoff_to_creator] + text_lc_tools if t is not None]

    print(f"\n  📦 最终 planner_tools 列表: {[t.name if hasattr(t, 'name') else 'unnamed' for t in planner_tools]}")
    print(f"  🎯 总工具数量: {len(planner_tools)} 个")
    print(f"{'='*80}\n")

    # 动态注入 text tool 工作流说明
    planner_base_prompt = PlannerAgentConfig().system_prompt
    # 替换 prompt 中的工具名称为实际工具名称
    # 支持多种旧名称格式，确保替换成功
    for old_name in ['transfer_to_image_video_creator', 'transfer_to_creator']:
        planner_base_prompt = planner_base_prompt.replace(old_name, handoff_tool_name)
    if text_lc_tools:
        tool_names = ', '.join(t.name for t in text_lc_tools)
        planner_prompt = planner_base_prompt.replace(
            f'Your ONLY two tools are: write_plan and {handoff_tool_name}.',
            f'You have these tools: write_plan, {handoff_tool_name}, and text generation tools ({tool_names}).'
        ) + f"""

TEXT GENERATION WORKFLOW (mandatory when text tools are available):
Available text tools: {tool_names}

For tasks requiring scripts, storylines, marketing copy, character descriptions,
scene details, or other rich textual content — you MUST use a text tool:
  Step 1. Call the text generation tool with a detailed prompt for the content needed.
  Step 2. Take the FULL output returned — do NOT summarize, shorten, or paraphrase it.
  Step 3. Place that exact text as the `description` of the relevant write_plan step(s).
          The creator agent reads step descriptions verbatim — completeness is essential.
  Step 4. Call write_plan with the steps populated from the text tool output.
  Step 5. Call {handoff_tool_name}.

For simple tasks (e.g. "generate 1 image of a cat"), you may skip the text tool
and call write_plan directly.
"""
    else:
        planner_prompt = planner_base_prompt

    # 使用安全的 ToolNode（自动跳过无效的 tool_call）
    # 返回 (ToolNode, post_model_hook | None)
    planner_tool_node, planner_post_hook = _make_safe_tool_node(planner_tools)

    print(f"\n{'='*80}")
    print(f"🚀 创建 Planner Agent (create_react_agent):")
    print(f"{'─'*80}")
    print(f"  📛 name: 'planner'")
    print(f"  🧠 model: {type(planner_lm).__name__}")
    print(f"  🛠️  tools (ToolNode): {type(planner_tool_node).__name__}")
    print(f"      └─ 包装的工具数量: {len(planner_tools)} 个")
    print(f"  📝 prompt 长度: {len(planner_prompt)} 字符")
    print(f"  🔧 pre_model_hook: {_pre_model_hook.__name__}")
    print(f"  🔧 post_model_hook: {planner_post_hook.__name__ if planner_post_hook else 'None'}")
    print(f"{'='*80}\n")

    planner_agent = create_react_agent(
        name='planner',
        model=planner_lm,
        tools=planner_tool_node,
        prompt=planner_prompt,
        pre_model_hook=_pre_model_hook,
        post_model_hook=planner_post_hook,
    )

    print(f"✅ Planner Agent 创建成功\n")
    print(f"{'='*100}\n")

    return planner_agent, planner_prompt


def _build_creator_agent(
    orchestrator: Any,
    media_lc_tools: List[Any],
    creator_prompt: str,
) -> Any:
    """创建 Creator agent。"""

    print(f"\n{'='*100}")
    print(f"🏗️  构建 Creator Agent")
    print(f"{'─'*100}")

    print(f"\n{'='*80}")
    print(f"🔧 Creator 工具配置:")
    print(f"{'─'*80}")
    print(f"  📋 media_lc_tools 数量: {len(media_lc_tools)}")
    if media_lc_tools:
        print(f"  📦 media_lc_tools: {[t.name if hasattr(t, 'name') else 'unnamed' for t in media_lc_tools]}")
    else:
        print(f"  ⚠️  无 image/video 工具可用")
    print(f"{'='*80}\n")

    # 使用安全的 ToolNode（自动跳过无效的 tool_call）
    creator_tool_node, creator_post_hook = _make_safe_tool_node(media_lc_tools)

    print(f"\n{'='*80}")
    print(f"🚀 创建 Creator Agent (create_react_agent):")
    print(f"{'─'*80}")
    print(f"  📛 name: 'assistant'")
    print(f"  🧠 model: {type(orchestrator).__name__}")
    print(f"  🛠️  tools (ToolNode): {type(creator_tool_node).__name__}")
    print(f"      └─ 包装的工具数量: {len(media_lc_tools)} 个")
    print(f"  📝 prompt 长度: {len(creator_prompt)} 字符")
    print(f"  🔧 pre_model_hook: {_pre_model_hook.__name__}")
    print(f"  🔧 post_model_hook: {creator_post_hook.__name__ if creator_post_hook else 'None'}")
    print(f"{'='*80}\n")

    agent = create_react_agent(
        name='assistant',
        model=orchestrator,
        tools=creator_tool_node,
        prompt=creator_prompt,
        pre_model_hook=_pre_model_hook,
        post_model_hook=creator_post_hook,
    )

    print(f"✅ Creator Agent 创建成功\n")
    print(f"{'='*100}\n")

    return agent


async def _run_stream_with_retry(
    swarm: Any,
    model_messages: List[Dict[str, Any]],
    context: Dict[str, Any],
    session_id: str,
) -> None:
    """执行流式处理，对瞬时错误自动重试。"""
    max_retries = 3
    for attempt in range(max_retries):
        processor = StreamProcessor(
            session_id, db_service, send_to_websocket)  # type: ignore
        try:
            await processor.process_stream(swarm, model_messages, context)
            return  # success
        except Exception as e:
            can_retry = (
                _is_transient_error(str(e))
                and processor.chunks_received == 0
                and attempt < max_retries - 1
            )
            if can_retry:
                wait = 5 * (attempt + 1)
                retry_num = attempt + 1
                max_possible_retries = max_retries - 1
                print(f"⚠️ Transient error on attempt {retry_num}/{max_retries}, retrying in {wait}s: {e}")
                await send_to_websocket(session_id, {
                    'type': 'info',
                    'info': f'模型繁忙，正在重试 ({retry_num}/{max_possible_retries})...'
                })
                await asyncio.sleep(wait)
                continue
            await _handle_error(e, session_id)
            return


async def langgraph_multi_agent(
    messages: List[Dict[str, Any]],
    canvas_id: str,
    session_id: str,
    text_model: ModelInfo,
    tool_list: List[ToolInfoJson],
    system_prompt: Optional[str] = None
) -> None:
    """统一的多模型 agent 处理函数

    架构：
    - 内置模型（builtin_model，server 端 settings 配置）作为 react agent 的编排大脑
    - tool_list 可包含三类工具：
        * type='text'  → 外部 text model，包装为 text generation tool
        * type='image' → 图像生成工具（来自 tool_service）
        * type='video' → 视频生成工具（来自 tool_service）
    - text_model（旧参数，向后兼容）若存在也作为 text tool 注入
    - 若 builtin_model 未配置，fallback 到第一个 text tool 作为编排器

    Args:
        messages: 消息历史
        canvas_id: 画布ID
        session_id: 会话ID
        text_model: 文本模型配置（旧参数，向后兼容；新前端直接放入 tool_list）
        tool_list: 工具列表（现在包含 text/image/video 三类）
        system_prompt: 系统提示词
    """
    try:
        has_text_model = text_model and text_model.get('model')
        has_tools = tool_list and len(tool_list) > 0

        if not has_text_model and not has_tools:
            raise ValueError(
                "Either text_model or tool_list must be provided. "
                "Please select at least one model or tool."
            )

        print(f"📝 has_text_model: {has_text_model}, has_tools: {has_tools}")

        # 0. 修复消息历史
        fixed_messages = _fix_chat_history(messages)

        # 1. 获取内置编排模型
        orchestrator = _create_builtin_model()

        # 2. 收集并按类型分流工具
        text_lc_tools, media_lc_tools = await _collect_tools(tool_list, text_model, has_text_model)

        # 3. 确保有编排器（fallback 到 text tool）
        orchestrator, text_lc_tools = await _ensure_orchestrator(
            orchestrator, text_lc_tools, text_model, has_text_model, tool_list
        )
        if not orchestrator:
            raise ValueError(
                "No model available for orchestration. "
                "Please configure a built-in model in settings, or select a text model tool."
            )

        # 4. 构建 Creator prompt
        creator_prompt = _build_creator_prompt(tool_list, system_prompt)

        # 4b. 注入媒体清单（server 端解析，保证图片/视频顺序权威）
        input_media = _parse_input_media_from_messages(fixed_messages)
        if input_media:
            creator_prompt = creator_prompt + "\n\n" + _format_media_manifest(input_media)

        # 5. 决定是否使用 planner-creator 双 agent 模式
        # 只要前端传了 text tools，就用 planner；否则单 agent
        text_tools_in_list = [t for t in (tool_list or []) if t.get('type') == 'text']
        use_planner = bool(text_tools_in_list)

        if not media_lc_tools:
            print("⚠️ 无可用 image/video 工具，creator agent 将向用户报告")

        print(f"🔍 use_planner={use_planner} | text_tools_in_list={len(text_tools_in_list)} | system_prompt={'yes' if system_prompt else 'no'}")

        # 6. 创建 agents
        agents: List[Any] = []
        planner_prompt = None

        if use_planner:
            planner_agent, planner_prompt = _build_planner_agent(text_tools_in_list, text_lc_tools)
            agents.append(planner_agent)
            print("🗺️ 使用 planner-creator 双 agent 模式")
        else:
            print("🎨 使用 creator 单 agent 模式")

        creator_agent = _build_creator_agent(orchestrator, media_lc_tools, creator_prompt)
        agents.append(creator_agent)

        swarm = create_swarm(
            agents=agents,
            default_active_agent='planner' if use_planner else 'assistant',
        )

        # 7. 准备消息 + 执行
        context = {
            'canvas_id': canvas_id,
            'session_id': session_id,
            'tool_list': tool_list or [],
            'callbacks': [PromptLoggingCallbackHandler()],
        }
        model_messages = _expand_image_urls(fixed_messages)
        model_messages, token_count = _truncate_messages_by_context_window(model_messages)
        _log_payload_size(model_messages, token_count)

        # 打印 prompt 日志：只打印实际已知的消息
        if use_planner:
            # Planner 接收用户消息 → 打印完整 prompt
            _log_llm_prompt(model_messages, planner_prompt or '', agent_name='planner')
            # Creator 的消息来自 planner handoff，此时无法预知 → 只打印 system prompt
            print(f"\n{'='*100}")
            print(f"📝 [LLM PROMPT LOG] Agent: creator (messages unknown — from planner handoff)")
            print(f"{'='*100}")
            print(f"{'─'*60}")
            print(f"  SYSTEM PROMPT (前 3000 字符):")
            print(f"{'─'*60}")
            preview = creator_prompt if len(creator_prompt) <= 3000 else creator_prompt[:3000] + f"\n... [已截断，总长度 {len(creator_prompt)} 字符]"
            print(preview)
            print(f"{'='*100}\n")
        else:
            _log_llm_prompt(model_messages, creator_prompt, agent_name='creator')

        await _run_stream_with_retry(swarm, model_messages, context, session_id)

    except Exception as e:
        await _handle_error(e, session_id)


_TRANSIENT_PATTERNS = ('upstream_error', 'rate limit', 'rate_limit', 'model is busy', 'server is busy', 'overloaded', 'too many requests', 'service unavailable', 'try again', 'nonetype')

def _is_transient_error(err_str: str) -> bool:
    lower = err_str.lower()
    return any(p in lower for p in _TRANSIENT_PATTERNS)


async def _handle_error(error: Exception, session_id: str) -> None:
    """处理错误"""
    print('Error in langgraph_agent', error)
    tb_str = traceback.format_exc()
    print(f"Full traceback:\n{tb_str}")
    traceback.print_exc()

    err_str = str(error)
    # Transient model-busy errors — show a friendly retry prompt instead of raw internals
    if 'upstream_error' in err_str or 'busy' in err_str.lower() or 'rate limit' in err_str.lower():
        user_message = '模型当前繁忙，请稍后重试。'
    else:
        user_message = err_str

    await send_to_websocket(session_id, cast(Dict[str, Any], {
        'type': 'error',
        'error': user_message
    }))
