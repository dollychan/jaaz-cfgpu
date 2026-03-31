import { ToolInfo } from '@/api/model'
import { LLMConfig } from '@/types/types'
import { create } from 'zustand'

type ConfigsStore = {
  initCanvas: boolean
  setInitCanvas: (initCanvas: boolean) => void

  /** 所有可用工具（text / image / video 三类，来自 /api/list_tools） */
  allTools: ToolInfo[]
  setAllTools: (tools: ToolInfo[]) => void

  /** 用户当前选中的工具（同样包含三类） */
  selectedTools: ToolInfo[]
  setSelectedTools: (models: ToolInfo[]) => void

  showInstallDialog: boolean
  setShowInstallDialog: (show: boolean) => void

  showUpdateDialog: boolean
  setShowUpdateDialog: (show: boolean) => void

  showSettingsDialog: boolean
  setShowSettingsDialog: (show: boolean) => void

  showLoginDialog: boolean
  setShowLoginDialog: (show: boolean) => void

  providers: {
    [key: string]: LLMConfig
  }
  setProviders: (providers: { [key: string]: LLMConfig }) => void
}

const useConfigsStore = create<ConfigsStore>((set) => ({
  initCanvas: false,
  setInitCanvas: (initCanvas) => set({ initCanvas }),

  allTools: [],
  setAllTools: (tools) => set({ allTools: tools }),

  selectedTools: [],
  setSelectedTools: (tools) => set({ selectedTools: tools }),

  showInstallDialog: false,
  setShowInstallDialog: (show) => set({ showInstallDialog: show }),

  showUpdateDialog: false,
  setShowUpdateDialog: (show) => set({ showUpdateDialog: show }),

  showSettingsDialog: false,
  setShowSettingsDialog: (show) => set({ showSettingsDialog: show }),

  showLoginDialog: false,
  setShowLoginDialog: (show) => set({ showLoginDialog: show }),

  providers: {},
  setProviders: (providers) => set({ providers }),
}))

export default useConfigsStore
