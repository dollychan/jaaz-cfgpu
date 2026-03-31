import React, { useState } from 'react'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuCheckboxItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
  DropdownMenuLabel,
  DropdownMenuGroup,
} from '@/components/ui/dropdown-menu'
import { Switch } from '@/components/ui/switch'
import { Checkbox } from '@/components/ui/checkbox'
import { useTranslation } from 'react-i18next'
import { useConfigs } from '@/contexts/configs'
import { ToolInfo } from '@/api/model'
import { PROVIDER_NAME_MAPPING } from '@/constants'
import { ScrollArea } from '@/components/ui/scroll-area'

interface ModelSelectorV3Props {
  onModelToggle?: (modelId: string, checked: boolean) => void
  onAutoToggle?: (enabled: boolean) => void
}

const ModelSelectorV3: React.FC<ModelSelectorV3Props> = ({
  onModelToggle,
  onAutoToggle
}) => {
  const {
    selectedTools,
    setSelectedTools,
    allTools,
  } = useConfigs()

  const [activeTab, setActiveTab] = useState<'image' | 'video' | 'text'>('image')
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const { t } = useTranslation()

  // auto 模式：所有非 text 类工具都被选中
  const mediaTols = allTools.filter(t => t.type !== 'text')
  const initialAutoMode = mediaTols.length > 0 && mediaTols.every(t => selectedTools.some(s => s.id === t.id))
  const [autoMode, setAutoMode] = useState(initialAutoMode)

  // 按 provider 分组
  const groupByProvider = (tools: ToolInfo[]) => {
    const grouped: { [provider: string]: ToolInfo[] } = {}
    tools.forEach((tool) => {
      if (!grouped[tool.provider]) grouped[tool.provider] = []
      grouped[tool.provider].push(tool)
    })
    // Jaaz first
    const sorted = Object.entries(grouped).sort(([a], [b]) => {
      if (a === 'jaaz') return -1
      if (b === 'jaaz') return 1
      return a.localeCompare(b)
    })
    return Object.fromEntries(sorted)
  }

  // 按类型过滤并分组
  const getToolsByType = (type: 'image' | 'video' | 'text') => {
    return groupByProvider(allTools.filter(t => t.type === type))
  }

  const getCurrentModels = () => getToolsByType(activeTab)

  // modelKey = provider:id
  const toKey = (t: ToolInfo) => `${t.provider}:${t.id}`

  const isModelSelected = (modelKey: string) =>
    selectedTools.some(t => toKey(t) === modelKey)

  const getProviderDisplayInfo = (provider: string) => {
    const info = PROVIDER_NAME_MAPPING[provider]
    return { name: info?.name || provider, icon: info?.icon }
  }

  /** Text 类工具：单选（只允许一个 text tool 同时选中） */
  const handleTextToolClick = (modelKey: string) => {
    const alreadySelected = isModelSelected(modelKey)
    if (alreadySelected) {
      // 取消选中
      setSelectedTools(selectedTools.filter(t => toKey(t) !== modelKey))
      localStorage.setItem(
        'disabled_tool_ids',
        JSON.stringify([...allTools.filter(t => toKey(t) !== modelKey || !isModelSelected(toKey(t))).map(t => t.id)])
      )
    } else {
      // 选中该 text tool，并移除之前选中的所有 text tools
      const tool = allTools.find(t => toKey(t) === modelKey)
      if (!tool) return
      const newSelected = [
        ...selectedTools.filter(t => t.type !== 'text'),
        tool,
      ]
      setSelectedTools(newSelected)
      localStorage.setItem(
        'disabled_tool_ids',
        JSON.stringify(allTools.filter(t => !newSelected.includes(t)).map(t => t.id))
      )
      onModelToggle?.(modelKey, true)
    }
  }

  /** Image/Video 工具 click 处理 */
  const handleMediaToolClick = (modelKey: string) => {
    if (autoMode) {
      // Auto 模式下点击 → 切到非 auto，只选中该工具
      const tool = allTools.find(t => toKey(t) === modelKey)
      if (!tool) return
      const newSelected = [
        ...selectedTools.filter(t => t.type === 'text'),  // 保留已选的 text tools
        tool,
      ]
      setSelectedTools(newSelected)
      localStorage.setItem(
        'disabled_tool_ids',
        JSON.stringify(allTools.filter(t => !newSelected.includes(t)).map(t => t.id))
      )
      setAutoMode(false)
      onAutoToggle?.(false)
      onModelToggle?.(modelKey, true)
    } else {
      // 非 auto 模式：image/video 工具互斥（同一次只选一个 media tool）
      const alreadySelected = isModelSelected(modelKey)
      if (alreadySelected) {
        // 取消
        const newSelected = selectedTools.filter(t => toKey(t) !== modelKey)
        setSelectedTools(newSelected)
        localStorage.setItem(
          'disabled_tool_ids',
          JSON.stringify(allTools.filter(t => !newSelected.includes(t)).map(t => t.id))
        )
      } else {
        // 选中，移除之前选中的 image/video tools（text tools 保留）
        const tool = allTools.find(t => toKey(t) === modelKey)
        if (!tool) return
        const newSelected = [
          ...selectedTools.filter(t => t.type === 'text'),
          tool,
        ]
        setSelectedTools(newSelected)
        localStorage.setItem(
          'disabled_tool_ids',
          JSON.stringify(allTools.filter(t => !newSelected.includes(t)).map(t => t.id))
        )
        onModelToggle?.(modelKey, true)
      }
    }
  }

  const handleModelClick = (modelKey: string) => {
    if (activeTab === 'text') {
      handleTextToolClick(modelKey)
    } else {
      handleMediaToolClick(modelKey)
    }
  }

  const handleAutoToggle = (enabled: boolean) => {
    if (activeTab === 'text') return  // text 类不支持 auto 模式

    if (enabled) {
      // 开启 auto：选中所有 media tools，text tools 保持不变
      const mediaTools = allTools.filter(t => t.type !== 'text')
      const newSelected = [
        ...selectedTools.filter(t => t.type === 'text'),
        ...mediaTools,
      ]
      setSelectedTools(newSelected)
      localStorage.setItem('disabled_tool_ids', JSON.stringify(
        allTools.filter(t => !newSelected.includes(t)).map(t => t.id)
      ))
    } else {
      // 关闭 auto：只保留第一个 image tool + 当前 text tools
      const imageTools = allTools.filter(t => t.type === 'image')
      const firstImageTool = imageTools[0] ?? null
      const newSelected = [
        ...selectedTools.filter(t => t.type === 'text'),
        ...(firstImageTool ? [firstImageTool] : []),
      ]
      setSelectedTools(newSelected)
      localStorage.setItem('disabled_tool_ids', JSON.stringify(
        allTools.filter(t => !newSelected.includes(t)).map(t => t.id)
      ))
    }
    setAutoMode(enabled)
    onAutoToggle?.(enabled)
  }

  const tabs = [
    { id: 'image', label: t('chat:modelSelector.tabs.image') },
    { id: 'video', label: t('chat:modelSelector.tabs.video') },
    { id: 'text', label: t('chat:modelSelector.tabs.text') }
  ] as const

  return (
    <DropdownMenu open={dropdownOpen} onOpenChange={setDropdownOpen}>
      <DropdownMenuTrigger asChild>
        <Button
          size={'sm'}
          variant="outline"
          className={`w-fit max-w-[40%] justify-between overflow-hidden ${autoMode
            ? 'bg-background border-border text-muted-foreground'
            : 'text-primary border-green-200 bg-green-50'
            }`}
        >
          {autoMode ? (
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path stroke="none" d="M0 0h24v24H0z" fill="none" /><path d="M4 4m0 1a1 1 0 0 1 1 -1h4a1 1 0 0 1 1 1v4a1 1 0 0 1 -1 1h-4a1 1 0 0 1 -1 -1z" /><path d="M4 14m0 1a1 1 0 0 1 1 -1h4a1 1 0 0 1 1 1v4a1 1 0 0 1 -1 1h-4a1 1 0 0 1 -1 -1z" /><path d="M14 14m0 1a1 1 0 0 1 1 -1h4a1 1 0 0 1 1 1v4a1 1 0 0 1 -1 1h-4a1 1 0 0 1 -1 -1z" /><path d="M14 7l6 0" /><path d="M17 4l0 6" /></svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor" className="icon icon-tabler icons-tabler-filled icon-tabler-apps"><path stroke="none" d="M0 0h24v24H0z" fill="none" /><path d="M9 3h-4a2 2 0 0 0 -2 2v4a2 2 0 0 0 2 2h4a2 2 0 0 0 2 -2v-4a2 2 0 0 0 -2 -2z" /><path d="M9 13h-4a2 2 0 0 0 -2 2v4a2 2 0 0 0 2 2h4a2 2 0 0 0 2 -2v-4a2 2 0 0 0 -2 -2z" /><path d="M19 13h-4a2 2 0 0 0 -2 2v4a2 2 0 0 0 2 2h4a2 2 0 0 0 2 -2v-4a2 2 0 0 0 -2 -2z" /><path d="M17 3a1 1 0 0 1 .993 .883l.007 .117v2h2a1 1 0 0 1 .117 1.993l-.117 .007h-2v2a1 1 0 0 1 -1.993 .117l-.007 -.117v-2h-2a1 1 0 0 1 -.117 -1.993l.117 -.007h2v-2a1 1 0 0 1 1 -1z" /></svg>
          )}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-96 select-none">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-2 border-b">
          <div>{t('chat:modelSelector.title')}</div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">{t('chat:modelSelector.auto')}</span>
            <Switch
              checked={autoMode}
              onCheckedChange={handleAutoToggle}
              disabled={activeTab === 'text'}
            />
          </div>
        </div>

        {/* Tabs */}
        <div className="flex p-1 bg-muted rounded-lg mx-4 my-2">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 px-3 py-1 rounded-md text-sm font-medium transition-colors cursor-pointer ${activeTab === tab.id
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
                }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Models List */}
        <ScrollArea>
          <div className="max-h-80 h-80 px-4 pb-4 select-none">
            {Object.entries(getCurrentModels()).map(([provider, providerModels], index, array) => {
              const providerInfo = getProviderDisplayInfo(provider)
              const isLastGroup = index === array.length - 1
              return (
                <DropdownMenuGroup key={provider}>
                  <DropdownMenuLabel className="text-xs font-medium text-muted-foreground px-0 py-2">
                    <div className="flex items-center gap-2">
                      <img
                        src={providerInfo.icon}
                        alt={providerInfo.name}
                        className="w-4 h-4 rounded-full"
                      />
                      {providerInfo.name}
                    </div>
                  </DropdownMenuLabel>
                  {providerModels.map((tool: ToolInfo) => {
                    const modelKey = toKey(tool)
                    const modelName = tool.display_name || tool.id

                    return (
                      <div
                        key={modelKey}
                        className="flex items-center justify-between p-3 hover:bg-muted/50 transition-colors mb-2 cursor-pointer"
                        onClick={() => handleModelClick(modelKey)}
                      >
                        <div className="flex-1">
                          <div className="font-medium text-sm">{modelName}</div>
                        </div>
                        <Checkbox
                          checked={isModelSelected(modelKey)}
                          className={`ml-4 ${autoMode && activeTab !== 'text' ? 'opacity-50' : ''}`}
                          disabled={autoMode && activeTab !== 'text'}
                        />
                      </div>
                    )
                  })}
                  {!isLastGroup && <DropdownMenuSeparator className="my-2" />}
                </DropdownMenuGroup>
              )
            })}
          </div>
        </ScrollArea>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export default ModelSelectorV3
