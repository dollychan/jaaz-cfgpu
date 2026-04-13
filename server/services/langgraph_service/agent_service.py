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
        text_model: 文本模型配置（前端传入，用于planner）
        tool_list: 工具模型配置列表（图像或视频模型）
        system_prompt: 系统提示词
    """
    try:
        # 0. 修复消息历史
        fixed_messages = _fix_chat_history(messages)

        # 1. 决定是否使用planner：只有前端传了text_model才使用planner
        use_planner = bool(text_model and text_model.get('model'))

        # 2. 获取builtin_model配置（必须存在，用于creator agent）
        settings = settings_service.get_raw_settings()
        builtin_model = settings.get('builtin_model', {})
        if not builtin_model or not builtin_model.get('model'):
            raise ValueError(
                "builtin_model is required. Please configure builtin_model in settings.json with a valid model."
            )

        # 3. 构建creator model配置
        provider = builtin_model.get('provider', 'cfgpu')
        provider_config = config_service.app_config.get(provider, {})
        creator_model = {
            'provider': provider,
            'model': builtin_model.get('model', ''),
            'url': builtin_model.get('url') or provider_config.get('url', ''),
            'type': 'text',
        }
        print(f"📝 Creator model (builtin_model): {creator_model}")

        # 4. 创建智能体
        if use_planner:
            # 前端传了text_model，使用planner制定计划
            print("🗺️ 使用planner-creator双agent模式（前端提供了text_model）")
            print(f"📝 Planner model (frontend): {text_model}")

            # 创建planner使用的model实例
            planner_model_instance = _create_text_model(text_model)

            # 创建creator使用的model实例（builtin_model）
            creator_model_instance = _create_text_model(creator_model)

            # 先创建planner agent（使用text_model）
            planner_agent = AgentManager.create_agents(
                planner_model_instance,
                [],  # planner不需要media tools
                system_prompt or "",
                tools_only=False
            )[0]  # 取第一个agent（planner）

            # 再创建creator agent（使用builtin_model）
            creator_agent = AgentManager.create_agents(
                creator_model_instance,
                tool_list,
                system_prompt or "",
                tools_only=True
            )[0]  # 取第一个agent（creator）

            agents = [planner_agent, creator_agent]
        else:
            # 前端没传text_model，直接使用creator
            print("🎨 直接使用creator agent模式（前端未提供text_model）")
            creator_model_instance = _create_text_model(creator_model)
            agents = AgentManager.create_agents(
                creator_model_instance,
                tool_list,
                system_prompt or "",
                tools_only=True
            )

        agent_names = [agent.name for agent in agents]
        print('👇agent_names', agent_names)
        last_agent = AgentManager.get_last_active_agent(
            fixed_messages, agent_names)

        print('👇last_agent', last_agent)

        # 5. 创建智能体群组
        swarm = create_swarm(
            agents=agents,  # type: ignore
            default_active_agent=last_agent if last_agent else agent_names[0]
        )

        # 6. 创建上下文
        context = {
            'canvas_id': canvas_id,
            'session_id': session_id,
            'tool_list': tool_list,
        }

        # 7. 流处理
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