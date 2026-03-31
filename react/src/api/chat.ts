import { Message, Model } from '@/types/types'
import { ModelInfo, ToolInfo } from './model'

export const getChatSession = async (sessionId: string) => {
  const response = await fetch(`/api/chat_session/${sessionId}`)
  if (!response.ok) return [] as Message[]
  const data = await response.json()
  return data as Message[]
}

export const sendMessages = async (payload: {
  sessionId: string
  canvasId: string
  newMessages: Message[]
  /** 向后兼容保留，新版前端通过 toolList 中 type='text' 的工具传递 text model */
  textModel?: Model
  toolList: ToolInfo[]
  systemPrompt: string | null
}) => {
  const response = await fetch(`/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      messages: payload.newMessages,
      canvas_id: payload.canvasId,
      session_id: payload.sessionId,
      text_model: payload.textModel,
      tool_list: payload.toolList,
      system_prompt: payload.systemPrompt,
    }),
  })
  if (!response.ok) {
    const err = await response.text()
    throw new Error(`Send messages failed (${response.status}): ${err}`)
  }
  const data = await response.json()
  return data as Message[]
}

export const cancelChat = async (sessionId: string) => {
  const response = await fetch(`/api/cancel/${sessionId}`, {
    method: 'POST',
  })
  if (!response.ok) return
  return await response.json()
}
