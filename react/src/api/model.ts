export type ModelInfo = {
  provider: string
  model: string
  type: 'text' | 'image' | 'tool' | 'video'
  url: string
}

export type ToolInfo = {
  provider: string
  /** image/video 工具的唯一 id；text 类型工具的 id 即为 model name */
  id: string
  display_name?: string | null
  /** text: 外部 LLM（用于复杂文本任务）; image/video: 媒体生成工具 */
  type?: 'image' | 'tool' | 'video' | 'text'
}

export async function listModels(): Promise<{
  llm: ModelInfo[]
  tools: ToolInfo[]
}> {
  // llm list 仅供 builtin_model 配置页面使用，日常选择全部走 list_tools
  const modelsResp = await fetch('/api/list_models')
    .then((res) => res.json())
    .catch((err) => {
      console.error(err)
      return []
    })
  const toolsResp = await fetch('/api/list_tools')
    .then((res) => res.json())
    .catch((err) => {
      console.error(err)
      return []
    })

  return {
    llm: modelsResp,
    tools: toolsResp,  // 现在包含 text / image / video 三类工具
  }
}
