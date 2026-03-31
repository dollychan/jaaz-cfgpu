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

        # Case 3: Only tools, no text model
        if has_tools and not has_text_model:
            print("🛠️ Mode: Tool-only execution (no text model)")
            # Create only the ImageVideoCreatorAgent
            agents = AgentManager.create_agents(
                model=None,  # No LLM model
                tool_list=tool_list,
                system_prompt=system_prompt or "",
                tools_only=True
            )

            # 0. 修复消息历史
            fixed_messages = _fix_chat_history(messages)
            agent_names = [agent.name for agent in agents]
            print('👇agent_names', agent_names)

            # 4. 创建智能体群组
            swarm = create_swarm(
                agents=agents,  # type: ignore
                default_active_agent=agent_names[0]
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
