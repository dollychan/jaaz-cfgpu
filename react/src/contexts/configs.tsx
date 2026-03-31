import { listModels, ToolInfo } from '@/api/model'
import useConfigsStore from '@/stores/configs'
import { useQuery } from '@tanstack/react-query'
import { createContext, useContext, useEffect, useRef } from 'react'

export const ConfigsContext = createContext<{
  configsStore: typeof useConfigsStore
  refreshModels: () => void
} | null>(null)

export const ConfigsProvider = ({
  children,
}: {
  children: React.ReactNode
}) => {
  const configsStore = useConfigsStore()
  const {
    setSelectedTools,
    setAllTools,
    setShowLoginDialog,
  } = configsStore

  const { data: modelList, refetch: refreshModels } = useQuery({
    queryKey: ['list_models_2'],
    queryFn: () => listModels(),
    staleTime: 1000,
    placeholderData: (previousData) => previousData,
    refetchOnWindowFocus: true,
    refetchOnReconnect: true,
    refetchOnMount: true,
  })

  // Issue 3.5: 用 ref 记录上次 toolList 的 id 集合签名，避免 window focus refetch
  // 每次都重算 selectedTools（覆盖用户在内存中的操作）
  const prevToolsKeyRef = useRef<string>('')

  useEffect(() => {
    if (!modelList) return
    // tools 现在包含 text / image / video 三类
    const { tools: toolList = [] } = modelList

    // 仅在 tool 列表实际发生变化时重算 selectedTools
    const toolsKey = JSON.stringify(toolList.map(t => t.id).sort())
    setAllTools(toolList)
    if (toolsKey === prevToolsKeyRef.current) return
    prevToolsKeyRef.current = toolsKey

    // 恢复用户上次的 disabled 工具偏好
    const disabledToolsJson = localStorage.getItem('disabled_tool_ids')
    let currentSelectedTools: ToolInfo[] = []
    if (disabledToolsJson) {
      try {
        const disabledToolIds: string[] = JSON.parse(disabledToolsJson)
        currentSelectedTools = toolList.filter(
          (t) => !disabledToolIds.includes(t.id)
        )
      } catch (error) {
        console.error(error)
      }
    } else {
      // 默认只选中 media tools（image/video），不自动选中 text tools
      currentSelectedTools = toolList.filter(t => t.type !== 'text')
      // 保存默认选择到 localStorage
      localStorage.setItem(
        'disabled_tool_ids',
        JSON.stringify(toolList.filter(t => t.type === 'text').map(t => t.id))
      )
    }

    setSelectedTools(currentSelectedTools)
  }, [
    modelList,
    setSelectedTools,
    setAllTools,
    setShowLoginDialog,
  ])

  return (
    <ConfigsContext.Provider
      value={{ configsStore: useConfigsStore, refreshModels }}
    >
      {children}
    </ConfigsContext.Provider>
  )
}

export const useConfigs = () => {
  const context = useContext(ConfigsContext)
  if (!context) {
    throw new Error('useConfigs must be used within a ConfigsProvider')
  }
  return context.configsStore()
}

export const useRefreshModels = () => {
  const context = useContext(ConfigsContext)
  if (!context) {
    throw new Error('useRefreshModels must be used within a ConfigsProvider')
  }
  return context.refreshModels
}
