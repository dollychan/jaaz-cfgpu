# Import service modules
from models.tool_model import ToolInfoJson
from services.db_service import db_service
from .StreamProcessor import StreamProcessor
from .agent_manager import AgentManager
import traceback
from utils.http_client import HttpClient
from langgraph_swarm import create_swarm  # type: ignore
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from services.websocket_service import send_to_websocket  # type: ignore
from services.config_service import config_service
from services.settings_service import settings_service
from typing import Optional, List, Dict, Any, cast, Set, TypedDict
from models.config_model import ModelInfo


class ContextInfo(TypedDict):
    """Context information passed to tools"""
    canvas_id: str
    session_id: str
    model_info: Dict[str, List[ModelInfo]]


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

    print(f"\n{'='*80}")
    print(f"🔧 修复聊天历史中的 tool_calls")
    print(f"{'─'*80}")
    print(f"  📋 输入消息数: {len(messages)}")

    fixed_messages: List[Dict[str, Any]] = []
    tool_call_ids: Set[str] = set()

    # 第一遍：收集所有ToolMessage的tool_call_id
    for msg in messages:
        if msg.get('role') == 'tool' and msg.get('tool_call_id'):
            tool_call_id = msg.get('tool_call_id')
            if tool_call_id:
                tool_call_ids.add(tool_call_id)

    print(f"  📦 收集到的 ToolMessage IDs: {len(tool_call_ids)} 个")

    # 第二遍：修复AIMessage中的tool_calls
    total_removed = 0
    total_fixed_args = 0
    for msg in messages:
        if msg.get('role') == 'assistant' and msg.get('tool_calls'):
            valid_tool_calls: List[Dict[str, Any]] = []
            removed_calls: List[str] = []
            fixed_args_calls: List[str] = []

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

                # 修复空 type（关键修复）
                if not tc_type or tc_type.strip() == '':
                    tool_call['type'] = 'function'
                    tc_type = 'function'

                if not fn or not fn.get('name'):
                    removed_calls.append(tool_call_id or '(no_id)')
                    continue

                # Qwen/Dashscope requires function.arguments to be a non-null string.
                # Fix any tool_calls with missing or null arguments.
                if fn is None:
                    fn = {}
                    tool_call['function'] = fn
                # 确保 arguments 存在且是有效的 JSON 字符串
                args = fn.get('arguments')
                if args is None or args == '' or not isinstance(args, str):
                    fn['arguments'] = '{}'
                    fixed_args_calls.append(fn.get('name', 'unknown'))
                    total_fixed_args += 1

                valid_tool_calls.append(tool_call)

            has_content = bool(msg.get('content'))

            if removed_calls:
                print(f"  ❌ 移除无效 tool_calls: {removed_calls}")
                total_removed += len(removed_calls)

            if fixed_args_calls:
                print(f"  🔧 修复 arguments: {fixed_args_calls}")

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

    print(f"\n  📊 修复统计:")
    print(f"     - 移除无效 tool_calls: {total_removed} 个")
    print(f"     - 修复 arguments 字段: {total_fixed_args} 个")
    print(f"     - 输出消息数: {len(fixed_messages)} 条")
    print(f"{'='*80}\n")

    return fixed_messages


async def langgraph_multi_agent(
    messages: List[Dict[str, Any]],
    canvas_id: str,
    session_id: str,
    text_model: ModelInfo,
    tool_list: List[ToolInfoJson],
    system_prompt: Optional[str] = None
) -> None:
    """多智能体处理函数

    Args:
        messages: 消息历史
        canvas_id: 画布ID
        session_id: 会话ID
        text_model: 文本模型配置
        tool_list: 工具模型配置列表（图像或视频模型）
        system_prompt: 系统提示词
    """
    try:
        # 0. 修复消息历史
        fixed_messages = _fix_chat_history(messages)

        # 1. 确定使用的模型：优先使用text_model，然后builtin_model，最后fallback到tool_list中的text工具
        effective_model = text_model
        if not (text_model and text_model.get('model')):
            # 尝试builtin_model
            builtin_model = settings_service.app_settings.get('builtin_model', {})
            if builtin_model and builtin_model.get('model'):
                # builtin_model配置了model，使用它
                provider = builtin_model.get('provider', 'cfgpu')
                provider_config = config_service.app_config.get(provider, {})

                effective_model = {
                    'provider': provider,
                    'model': builtin_model.get('model', ''),
                    'url': builtin_model.get('url') or provider_config.get('url', ''),
                    'type': 'text',
                }
                print(f"⚠️ text_model为空，使用builtin_model: {effective_model}")
            else:
                # Fallback: 从tool_list中找第一个text类型的工具作为模型
                text_tools = [t for t in (tool_list or []) if t.get('type') == 'text']
                if text_tools:
                    first_text_tool = text_tools[0]
                    provider = first_text_tool.get('provider', '')
                    # 从config_service获取provider的URL配置
                    provider_config = config_service.app_config.get(provider, {})
                    url = provider_config.get('url', '')

                    effective_model = {
                        'provider': provider,
                        'model': first_text_tool.get('id', ''),
                        'url': url,
                        'type': 'text',
                    }
                    print(f"⚠️ text_model和builtin_model都为空，使用tool_list fallback: {effective_model}")
                else:
                    raise ValueError(
                        "No text model available. Please provide text_model, configure builtin_model in settings, or include a text tool in tool_list."
                    )

        # 2. 文本模型
        text_model_instance = _create_text_model(effective_model)

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
