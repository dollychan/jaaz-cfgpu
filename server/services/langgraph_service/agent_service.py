from models.tool_model import ToolInfoJson
from services.db_service import db_service
from .StreamProcessor import StreamProcessor
from .agent_manager import AgentManager
import traceback
import re
import json
import uuid
from utils.http_client import HttpClient
from langgraph_swarm import create_swarm  # type: ignore
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from services.websocket_service import send_to_websocket  # type: ignore
from services.config_service import config_service
from services.tool_service import tool_service
from typing import Optional, List, Dict, Any, cast, Set, TypedDict
from models.config_model import ModelInfo


class ContextInfo(TypedDict):
    """Context information passed to tools"""
    canvas_id: str
    session_id: str
    model_info: Dict[str, List[ModelInfo]]


def _fix_chat_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """修复聊天历史中不完整的工具调用

    根据LangGraph文档建议，移除没有对应ToolMessage的tool_calls
    参考: https://langchain-ai.github.io/langgraph/troubleshooting/errors/INVALID_CHAT_HISTORY/
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
            # 过滤掉没有对应ToolMessage的tool_calls
            valid_tool_calls: List[Dict[str, Any]] = []
            removed_calls: List[str] = []

            for tool_call in msg.get('tool_calls', []):
                tool_call_id = tool_call.get('id')
                if tool_call_id in tool_call_ids:
                    valid_tool_calls.append(tool_call)
                elif tool_call_id:
                    removed_calls.append(tool_call_id)

            # 记录修复信息
            if removed_calls:
                print(
                    f"🔧 修复消息历史：移除了 {len(removed_calls)} 个不完整的工具调用: {removed_calls}")

            # 更新消息
            if valid_tool_calls:
                msg_copy = msg.copy()
                msg_copy['tool_calls'] = valid_tool_calls
                fixed_messages.append(msg_copy)
            elif msg.get('content'):  # 如果没有有效的tool_calls但有content，保留消息
                msg_copy = msg.copy()
                msg_copy.pop('tool_calls', None)  # 移除空的tool_calls
                fixed_messages.append(msg_copy)
            # 如果既没有有效tool_calls也没有content，跳过这条消息
        else:
            # 非assistant消息或没有tool_calls的消息直接保留
            fixed_messages.append(msg)

    return fixed_messages


def _get_last_user_text(messages: List[Dict[str, Any]]) -> str:
    """从消息列表中取出最后一条用户消息的纯文本内容"""
    for msg in reversed(messages):
        if msg.get('role') == 'user':
            content = msg.get('content', '')
            if isinstance(content, list):
                # 多模态消息：拼接所有 text 部分
                return ' '.join(
                    part.get('text', '')
                    for part in content
                    if isinstance(part, dict) and part.get('type') == 'text'
                )
            return str(content)
    return ''


def _parse_tool_params_from_message(message: str) -> Dict[str, Any]:
    """从用户消息中解析工具调用参数

    支持解析：
    - <input_images><image file_id="xxx"/></input_images>
    - <input_videos><video file_id="xxx"/></input_videos>
    - <input_audios><audio file_id="xxx"/></input_audios>
    - <duration>10</duration>
    - <aspect_ratio>16:9</aspect_ratio>
    - CFGPU asset JSON 结构：{"type": "image_url", "image_url": {"url": "asset://..."}}
    - 剩余文本作为 prompt
    """
    params: Dict[str, Any] = {}
    remaining = message

    # --- XML 标签解析 ---
    def extract_xml_file_ids(tag: str, inner_tag: str) -> List[str]:
        nonlocal remaining
        match = re.search(rf'<{tag}>(.*?)</{tag}>', remaining, re.DOTALL)
        if not match:
            return []
        inner = match.group(1)
        file_ids = re.findall(rf'<{inner_tag}[^>]+file_id=["\']([^"\']+)["\']', inner)
        remaining = remaining.replace(match.group(0), '')
        return file_ids

    images = extract_xml_file_ids('input_images', 'image')
    if images:
        params['input_images'] = images

    videos = extract_xml_file_ids('input_videos', 'video')
    if videos:
        params['input_videos'] = videos

    audios = extract_xml_file_ids('input_audios', 'audio')
    if audios:
        params['input_audios'] = audios

    # <duration>
    duration_match = re.search(r'<duration>(\d+)</duration>', remaining)
    if duration_match:
        params['duration'] = int(duration_match.group(1))
        remaining = remaining.replace(duration_match.group(0), '')

    # <aspect_ratio>
    ar_match = re.search(r'<aspect_ratio>([^<]+)</aspect_ratio>', remaining)
    if ar_match:
        params['aspect_ratio'] = ar_match.group(1).strip()
        remaining = remaining.replace(ar_match.group(0), '')

    # --- CFGPU asset JSON 结构解析 ---
    # 匹配形如：
    #   {
    #     "type": "image_url",
    #     "image_url": {"url": "asset://xxx"},
    #     "role": "reference_image"
    #   }
    # 外层 {} 内部可能包含一层嵌套 {}（image_url/video_url/audio_url 的值对象）
    # 使用允许一层嵌套的正则：(?:[^{}]|\{[^{}]*\})*
    asset_block_pattern = re.compile(
        r'\{(?:[^{}]|\{[^{}]*\})*?"type"\s*:\s*"(image_url|video_url|audio_url)"(?:[^{}]|\{[^{}]*\})*?\}',
        re.DOTALL
    )
    for block_match in list(asset_block_pattern.finditer(remaining)):
        block_str = block_match.group(0)
        url_type = block_match.group(1)
        # 从 block 中提取 asset:// URL
        url_match = re.search(r'"url"\s*:\s*"(asset://[^"]+)"', block_str)
        if url_match:
            asset_url = url_match.group(1)
            if url_type == 'image_url':
                params.setdefault('input_images', []).append(asset_url)
            elif url_type == 'video_url':
                params.setdefault('input_videos', []).append(asset_url)
            elif url_type == 'audio_url':
                params.setdefault('input_audios', []).append(asset_url)
            remaining = remaining.replace(block_str, '')

    # 剩余文本清理后作为 prompt
    prompt = remaining.strip()
    params['prompt'] = prompt

    return params


async def _execute_tools_directly(
    messages: List[Dict[str, Any]],
    canvas_id: str,
    session_id: str,
    tool_list: List[ToolInfoJson],
) -> None:
    """无 text model 时直接执行工具

    解析用户消息中的参数（prompt / input_images / input_videos / input_audios /
    duration / aspect_ratio），然后对 tool_list 中的每个工具依次调用。
    """
    user_text = _get_last_user_text(messages)
    params = _parse_tool_params_from_message(user_text)

    print(f"🛠️ Direct tool execution params: {params}")

    runnable_config = {
        'configurable': {
            'canvas_id': canvas_id,
            'session_id': session_id,
            'tool_list': tool_list,
        }
    }

    # 收集本次执行产生的所有消息，用于保存到 DB
    assistant_tool_calls = []
    tool_result_messages = []

    for tool_json in tool_list:
        tool_id = tool_json.get('id', '')
        tool_fn = tool_service.get_tool(tool_id)
        if not tool_fn:
            print(f"⚠️ Tool not found: {tool_id}")
            continue

        # 根据工具的 schema 过滤参数，只传工具实际接受的字段
        tool_args: Dict[str, Any] = {}
        if hasattr(tool_fn, 'args_schema') and tool_fn.args_schema:
            accepted_fields = set(tool_fn.args_schema.model_fields.keys())
            for k, v in params.items():
                if k in accepted_fields:
                    tool_args[k] = v
        else:
            # 没有 schema 信息时，只传 prompt
            if params.get('prompt'):
                tool_args['prompt'] = params['prompt']

        # prompt 是必需参数，如果为空则跳过
        if not tool_args.get('prompt'):
            print(f"⚠️ Skipping tool {tool_id}: no prompt available")
            continue

        # 生成 tool_call_id
        call_id = str(uuid.uuid4())
        tool_args['tool_call_id'] = call_id

        # 通知前端工具调用开始
        await send_to_websocket(session_id, {
            'type': 'tool_call',
            'id': call_id,
            'name': tool_id,
            'arguments': json.dumps({k: v for k, v in tool_args.items() if k != 'tool_call_id'}),
        })

        # 记录 assistant tool_call 信息
        assistant_tool_calls.append({
            'id': call_id,
            'type': 'function',
            'function': {
                'name': tool_id,
                'arguments': json.dumps({k: v for k, v in tool_args.items() if k != 'tool_call_id'}),
            }
        })

        try:
            # 含 InjectedToolCallId 的工具必须以 ToolCall 格式调用：
            # {'args': {...}, 'name': '...', 'type': 'tool_call', 'tool_call_id': '...'}
            tool_call_input = {
                'args': {k: v for k, v in tool_args.items() if k != 'tool_call_id'},
                'name': tool_id,
                'type': 'tool_call',
                'id': call_id,  # LangChain ToolCall TypedDict 使用 'id' 而非 'tool_call_id'
            }
            result = await tool_fn.ainvoke(tool_call_input, config=runnable_config)
            tool_result_content = str(result)
        except Exception as e:
            tool_result_content = f"Tool execution failed: {str(e)}"
            print(f"❌ Tool {tool_id} failed: {e}")
            traceback.print_exc()

        tool_result_msg = {
            'role': 'tool',
            'tool_call_id': call_id,
            'content': tool_result_content,
        }
        tool_result_messages.append(tool_result_msg)

        # 通知前端工具调用结果
        await send_to_websocket(session_id, {
            'type': 'tool_call_result',
            'id': call_id,
            'message': tool_result_msg,
        })

    # 保存消息到 DB：assistant 消息（含 tool_calls）+ 每个 tool 结果消息
    if assistant_tool_calls:
        assistant_msg = {
            'role': 'assistant',
            'content': None,
            'tool_calls': assistant_tool_calls,
        }
        await db_service.create_message(session_id, 'assistant', json.dumps(assistant_msg))

    for tool_msg in tool_result_messages:
        await db_service.create_message(session_id, 'tool', json.dumps(tool_msg))

    # 发送全量消息到前端（供前端同步状态）
    all_messages_snapshot = list(messages) + (
        [{'role': 'assistant', 'content': None, 'tool_calls': assistant_tool_calls}]
        if assistant_tool_calls else []
    ) + tool_result_messages
    await send_to_websocket(session_id, {
        'type': 'all_messages',
        'messages': all_messages_snapshot,
    })


async def langgraph_multi_agent(
    messages: List[Dict[str, Any]],
    canvas_id: str,
    session_id: str,
    text_model: ModelInfo,
    tool_list: List[ToolInfoJson],
    system_prompt: Optional[str] = None
) -> None:
    """多智能体处理函数

    根据text_model和tool_list的有无决定处理方式：
    - 有text_model + 有工具 → planner + creator agents（原来的逻辑）
    - 只有text_model → 直接使用text model处理（不需要agents）
    - 只有工具 → 只创建creator agent处理
    - 都没有 → 错误

    Args:
        messages: 消息历史
        canvas_id: 画布ID
        session_id: 会话ID
        text_model: 文本模型配置（可能为空）
        tool_list: 工具模型配置列表（可能为空）
        system_prompt: 系统提示词
    """
    try:
        # Check what we have to work with
        has_text_model = text_model and text_model.get('model')
        has_tools = tool_list and len(tool_list) > 0

        print(f"📝 has_text_model: {has_text_model}, has_tools: {has_tools}")

        # Case 1: Only text model, no tools → direct LLM call
        if has_text_model and not has_tools:
            print("💬 Mode: Direct text model (no tools)")
            text_model_instance = _create_text_model(text_model)
            fixed_messages = _fix_chat_history(messages)

            # Convert messages to LangChain format
            from langchain_core.messages import (
                HumanMessage,
                AIMessage,
                ToolMessage,
                SystemMessage,
            )

            lc_messages = []
            for msg in fixed_messages:
                role = msg.get('role')
                content = msg.get('content', '')

                if role == 'user':
                    lc_messages.append(HumanMessage(content=content))
                elif role == 'assistant':
                    lc_messages.append(AIMessage(content=content))
                elif role == 'system':
                    lc_messages.append(SystemMessage(content=content))
                elif role == 'tool':
                    lc_messages.append(ToolMessage(
                        content=content,
                        tool_call_id=msg.get('tool_call_id', '')
                    ))

            # If system prompt provided, prepend it
            if system_prompt:
                lc_messages.insert(0, SystemMessage(content=system_prompt))

            # Call LLM
            response = await text_model_instance.ainvoke(lc_messages)

            # Send response
            await send_to_websocket(session_id, {
                'type': 'delta',
                'delta': response.content,
                'role': 'assistant'
            })
            return

        # Case 2: Both text model and tools (original multi-agent flow)
        if has_text_model and has_tools:
            print("🤖 Mode: Multi-agent (text model + tools)")
            # 0. 修复消息历史
            fixed_messages = _fix_chat_history(messages)

            # 2. 文本模型
            text_model_instance = _create_text_model(text_model)

            # 3. 创建智能体
            agents = AgentManager.create_agents(
                text_model_instance,
                tool_list,  # 传入所有注册的工具
                system_prompt or ""
            )
            agent_names = [agent.name for agent in agents]
            print('👇agent_names', agent_names)
            last_agent = AgentManager.get_last_active_agent(
                fixed_messages, agent_names)

            print('👇last_agent', last_agent)

            # 4. 创建智能体群组
            swarm = create_swarm(
                agents=agents,  # type: ignore
                default_active_agent=last_agent if last_agent else agent_names[0]
            )

            # 5. 创建上下文
            context = {
                'canvas_id': canvas_id,
                'session_id': session_id,
                'tool_list': tool_list,
            }

            # 6. 流处理
            processor = StreamProcessor(
                session_id, db_service, send_to_websocket)  # type: ignore
            await processor.process_stream(swarm, fixed_messages, context)
            return

        # Case 3: Only tools, no text model — directly invoke tools without LLM
        if has_tools and not has_text_model:
            print("🛠️ Mode: Direct tool execution (no text model)")
            fixed_messages = _fix_chat_history(messages)
            await _execute_tools_directly(fixed_messages, canvas_id, session_id, tool_list)
            return

        # Case 4: Neither text model nor tools
        raise ValueError("Either text_model or tool_list must be provided")

    except Exception as e:
        await _handle_error(e, session_id)


def _create_text_model(text_model: ModelInfo) -> Any:
    """创建语言模型实例"""
    model = text_model.get('model')
    provider = text_model.get('provider')
    url = text_model.get('url')
    api_key = config_service.app_config.get(  # type: ignore
        provider, {}).get("api_key", "")

    # TODO: Verify if max token is working
    # max_tokens = text_model.get('max_tokens', 8148)

    if provider == 'ollama':
        return ChatOllama(
            model=model,
            base_url=url,
        )
    else:
        # Create httpx client with SSL configuration for ChatOpenAI
        http_client = HttpClient.create_sync_client()
        http_async_client = HttpClient.create_async_client()
        return ChatOpenAI(
            model=model,
            api_key=api_key,  # type: ignore
            timeout=300,
            base_url=url,
            temperature=0,
            # max_tokens=max_tokens, # TODO: 暂时注释掉有问题的参数
            http_client=http_client,
            http_async_client=http_async_client
        )


async def _handle_error(error: Exception, session_id: str) -> None:
    """处理错误"""
    print('Error in langgraph_agent', error)
    tb_str = traceback.format_exc()
    print(f"Full traceback:\n{tb_str}")
    traceback.print_exc()

    await send_to_websocket(session_id, cast(Dict[str, Any], {
        'type': 'error',
        'error': str(error)
    }))
