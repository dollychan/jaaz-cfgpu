from models.tool_model import ToolInfoJson
from services.db_service import db_service
from .StreamProcessor import StreamProcessor
from .configs import PlannerAgentConfig, create_handoff_tool
import traceback
import json
import os
from utils.http_client import HttpClient
from langgraph_swarm import create_swarm  # type: ignore
from langgraph.prebuilt import create_react_agent  # type: ignore
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool as lc_tool  # type: ignore
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
            valid_tool_calls: List[Dict[str, Any]] = []
            removed_calls: List[str] = []

            for tool_call in msg.get('tool_calls', []):
                tool_call_id = tool_call.get('id')
                if tool_call_id in tool_call_ids:
                    valid_tool_calls.append(tool_call)
                elif tool_call_id:
                    removed_calls.append(tool_call_id)

            if removed_calls:
                print(
                    f"🔧 修复消息历史：移除了 {len(removed_calls)} 个不完整的工具调用: {removed_calls}")

            if valid_tool_calls:
                msg_copy = msg.copy()
                msg_copy['tool_calls'] = valid_tool_calls
                fixed_messages.append(msg_copy)
            elif msg.get('content'):
                msg_copy = msg.copy()
                msg_copy.pop('tool_calls', None)
                fixed_messages.append(msg_copy)
        else:
            fixed_messages.append(msg)

    return fixed_messages


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

        # 2. 构建工具列表
        all_lc_tools: List[Any] = []

        # 向后兼容：旧前端传来的 text_model 也作为 text tool 注入
        if has_text_model:
            text_tool = _create_text_generation_tool(dict(text_model))
            if text_tool:
                all_lc_tools.append(text_tool)
                print(f"✅ 添加 text tool（来自 text_model）: {text_tool.name}")

        # 遍历 tool_list
        for tool_json in (tool_list or []):
            tool_type = tool_json.get('type', '')
            tool_id = tool_json.get('id', '')

            if tool_type == 'text':
                # text model → 包装成 text generation tool
                txt_tool = _create_text_generation_tool(dict(tool_json))
                if txt_tool:
                    all_lc_tools.append(txt_tool)
                    print(f"✅ 添加 text tool（来自 tool_list）: {txt_tool.name}")
            else:
                # image/video → 从 tool_service 获取
                lc_t = tool_service.get_tool(tool_id)
                if lc_t:
                    all_lc_tools.append(lc_t)
                else:
                    print(f"⚠️ 工具未找到: {tool_id}")

        print(f"🛠️ 可用工具列表: {[t.name for t in all_lc_tools]}")

        # 3. 如果没有内置模型，fallback 到第一个 text tool 作为编排器
        if not orchestrator:
            # 找到第一个 text generation tool，作为编排器（非工具）
            text_tools = [t for t in all_lc_tools if t.name.startswith('generate_text_with_')]
            media_tools = [t for t in all_lc_tools if not t.name.startswith('generate_text_with_')]

            if text_tools:
                # 取第一个 text tool 对应的 model info，创建 orchestrator
                if has_text_model:
                    orchestrator = _create_text_model(text_model)
                    all_lc_tools = media_tools  # text model 已作为 orchestrator，不再作为 tool
                    print(f"⚠️ builtin_model 未配置，使用 text_model 作为编排器（向后兼容模式）")
                else:
                    # 从 tool_list 找第一个 text type
                    first_text_tool_json = next(
                        (t for t in (tool_list or []) if t.get('type') == 'text'), None
                    )
                    if first_text_tool_json:
                        provider = first_text_tool_json.get('provider', '')
                        model_name = first_text_tool_json.get('id', '')
                        url = config_service.app_config.get(provider, {}).get('url', '')
                        lm: ModelInfo = {'provider': provider, 'model': model_name, 'url': url, 'type': 'text'}
                        orchestrator = _create_text_model(lm)
                        # 从 tool list 中移除这个 text tool（避免模型调用自己）
                        tool_name_to_remove = text_tools[0].name
                        all_lc_tools = [t for t in all_lc_tools if t.name != tool_name_to_remove]
                        print(f"⚠️ builtin_model 未配置，使用 {model_name} 作为编排器（fallback 模式）")

        if not orchestrator:
            raise ValueError(
                "No model available for orchestration. "
                "Please configure a built-in model in settings, or select a text model tool."
            )

        # 4. 构建 system prompt（使用 ImageVideoCreatorAgent 的 prompt，它包含丰富的输入检测规则）
        from .configs.image_vide_creator_config import ImageVideoCreatorAgentConfig
        creator_config = ImageVideoCreatorAgentConfig(tool_list or [])
        agent_system_prompt = system_prompt or creator_config.system_prompt

        # 5. 判断是否使用 planner-creator 双 agent 模式（Issue 4.2）
        # 条件：tool_list 中有 text tools，且没有自定义 system_prompt（自定义时走单 agent）
        text_tools_in_list = [t for t in (tool_list or []) if t.get('type') == 'text']
        use_planner = bool(text_tools_in_list and not system_prompt)

        if use_planner:
            # planner agent：使用第一个 text model 作为编排大脑，只负责制定计划并移交
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
            write_plan_lc = tool_service.get_tool('write_plan')
            handoff_to_creator = create_handoff_tool(
                agent_name='assistant',
                description='Transfer to the image/video creator agent to execute the plan.',
            )
            planner_tools = [t for t in [write_plan_lc, handoff_to_creator] if t is not None]
            planner_agent = create_react_agent(
                name='planner',
                model=planner_lm,
                tools=planner_tools,
                prompt=PlannerAgentConfig().system_prompt,
            )

            # creator agent：使用 builtin model，持有所有 image/video/text gen tools
            creator_agent = create_react_agent(
                name='assistant',
                model=orchestrator,
                tools=all_lc_tools,
                prompt=agent_system_prompt,
            )

            swarm = create_swarm(
                agents=[planner_agent, creator_agent],
                default_active_agent='planner',
            )
            print("🗺️ 使用 planner-creator 双 agent 模式")
        else:
            # 单 agent 模式：无 text tools 或自定义 system_prompt
            agent = create_react_agent(
                name='assistant',
                model=orchestrator,
                tools=all_lc_tools,
                prompt=agent_system_prompt,
            )
            swarm = create_swarm(
                agents=[agent],
                default_active_agent='assistant',
            )

        # 7. 创建上下文并运行
        context = {
            'canvas_id': canvas_id,
            'session_id': session_id,
            'tool_list': tool_list or [],
        }

        # 将相对图片 URL（/api/file/... 或 /api/material/serve/...）展开为
        # 外部可访问的绝对 URL，供远端模型 API（如 cfgpu）使用
        model_messages = _expand_image_urls(fixed_messages)
        model_messages, token_count = _truncate_messages_by_context_window(model_messages)
        _log_payload_size(model_messages, token_count)

        processor = StreamProcessor(
            session_id, db_service, send_to_websocket)  # type: ignore
        await processor.process_stream(swarm, model_messages, context)

    except Exception as e:
        await _handle_error(e, session_id)


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
