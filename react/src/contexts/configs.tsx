import { listModels, ToolInfo } from '@/api/model'
import useConfigsStore from '@/stores/configs'
import { useQuery } from '@tanstack/react-query'
import { createContext, useContext, useEffect } from 'react'

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

  useEffect(() => {
    if (!modelList) return
    // tools 现在包含 text / image / video 三类
    const { tools: toolList = [] } = modelList

    setAllTools(toolList)

    // 恢复用户上次的 disabled 工具偏好
    const disabledToolsJson = localStorage.getItem('disabled_tool_ids')
    let currentSelectedTools: ToolInfo[] = toolList  // 默认全选
    if (disabledToolsJson) {
      try {
        const disabledToolIds: string[] = JSON.parse(disabledToolsJson)
        currentSelectedTools = toolList.filter(
          (t) => !disabledToolIds.includes(t.id)
        )
      } catch (error) {
        console.error(error)
      }
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
