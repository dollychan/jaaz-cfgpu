import { cancelChat } from '@/api/chat'
import { cancelMagicGenerate } from '@/api/magic'
import { uploadImage, uploadVideo, uploadAudio } from '@/api/upload'
import { getMaterialFilesApi, AssetFileRecord } from '@/api/material'
import { Button } from '@/components/ui/button'
import { useConfigs } from '@/contexts/configs'
import {
  eventBus,
  TCanvasAddImagesToChatEvent,
  TMaterialAddImagesToChatEvent,
} from '@/lib/event'
import { cn, dataURLToFile } from '@/lib/utils'
import { Message, MessageContent, Model } from '@/types/types'
import { ModelInfo, ToolInfo } from '@/api/model'
import { useMutation, useQuery } from '@tanstack/react-query'
import { useDrop } from 'ahooks'
import { produce } from 'immer'
import {
  ArrowUp,
  Loader2,
  PlusIcon,
  Search,
  Square,
  XIcon,
  RectangleVertical,
  ChevronDown,
  Hash,
  Video,
  Music,
  Library,
} from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import Textarea, { TextAreaRef } from 'rc-textarea'
import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import ModelSelectorV3 from './ModelSelectorV3'
import { useAuth } from '@/contexts/AuthContext'
import { useBalance } from '@/hooks/use-balance'
import { BASE_API_URL } from '@/constants'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import { Input } from '@/components/ui/input'

type ChatTextareaProps = {
  pending: boolean
  className?: string
  messages: Message[]
  sessionId?: string
  onSendMessages: (
    data: Message[],
    configs: {
      toolList: ToolInfo[]
    }
  ) => void
  onCancelChat?: () => void
}

const ChatTextarea: React.FC<ChatTextareaProps> = ({
  pending,
  className,
  messages,
  sessionId,
  onSendMessages,
  onCancelChat,
}) => {
  const { t } = useTranslation()
  const { authStatus } = useAuth()
  const { selectedTools, setShowLoginDialog } = useConfigs()
  const { balance } = useBalance()
  const [prompt, setPrompt] = useState('')
  const textareaRef = useRef<TextAreaRef>(null)
  const [images, setImages] = useState<
    {
      file_id: string
      width: number
      height: number
      /** 'local' = uploaded to /api/file, 'asset' = from material library */
      type: 'local' | 'asset'
      /** For asset type: /api/material/serve/{disk_name} */
      serve_path?: string
    }[]
  >([])
  const [isFocused, setIsFocused] = useState(false)
  const [selectedAspectRatio, setSelectedAspectRatio] = useState<string>(
    () => localStorage.getItem('chat_aspect_ratio') ?? 'auto'
  )
  const [quantity, setQuantity] = useState<number>(
    () => Number(localStorage.getItem('chat_quantity') ?? '1')
  )
  const [showQuantitySlider, setShowQuantitySlider] = useState(false)
  const quantitySliderRef = useRef<HTMLDivElement>(null)
  const MAX_QUANTITY = 30
  const [duration, setDuration] = useState<number>(
    () => Number(localStorage.getItem('chat_duration') ?? '5')
  )
  const [showDurationSlider, setShowDurationSlider] = useState(false)
  const durationSliderRef = useRef<HTMLDivElement>(null)
  const MIN_DURATION = 4
  const MAX_DURATION = 15

  const [assetPickerOpen, setAssetPickerOpen] = useState(false)
  const [assetSearchQuery, setAssetSearchQuery] = useState('')

  // Material asset picker
  const { data: materialFiles = [] } = useQuery({
    queryKey: ['material-files'],
    queryFn: getMaterialFilesApi,
    staleTime: 30_000,
  })

  const fileInputRef = useRef<HTMLInputElement>(null)
  const [videos, setVideos] = useState<{
    file_id: string
    type: 'local' | 'asset'
    serve_path?: string
  }[]>([])
  const [audios, setAudios] = useState<{
    file_id: string
    type: 'local' | 'asset'
    serve_path?: string
  }[]>([])

  // 充值按钮组件
  const RechargeContent = useCallback(() => (
    <div className="flex items-center justify-between gap-3">
      <span className="text-sm text-muted-foreground flex-1">
        {t('chat:insufficientBalanceDescription')}
      </span>
      <Button
        size="sm"
        variant="outline"
        className="shrink-0"
        onClick={() => {
          const billingUrl = `${BASE_API_URL}/billing`
          if (window.electronAPI?.openBrowserUrl) {
            window.electronAPI.openBrowserUrl(billingUrl)
          } else {
            window.open(billingUrl, '_blank')
          }
        }}
      >
        {t('common:auth.recharge')}
      </Button>
    </div>
  ), [t])

  const { mutate: uploadImageMutation } = useMutation({
    mutationFn: (file: File) => uploadImage(file),
    onSuccess: (data) => {
      console.log('🦄uploadImageMutation onSuccess', data)
      setImages((prev) => [
        ...prev,
        {
          file_id: data.file_id,
          width: data.width,
          height: data.height,
          type: 'local',
        },
      ])
    },
    onError: (error) => {
      console.error('🦄uploadImageMutation onError', error)
      toast.error('Failed to upload image', {
        description: <div>{error.toString()}</div>,
      })
    },
  })

  const { mutate: uploadVideoMutation } = useMutation({
    mutationFn: (file: File) => uploadVideo(file),
    onSuccess: (data) => {
      setVideos((prev) => [...prev, { file_id: data.file_id, type: 'local' }])
    },
    onError: (error) => {
      console.error('🎥 uploadVideoMutation onError', error)
      toast.error('Failed to upload video', {
        description: <div>{error.toString()}</div>,
      })
    },
  })

  const { mutate: uploadAudioMutation } = useMutation({
    mutationFn: (file: File) => uploadAudio(file),
    onSuccess: (data) => {
      setAudios((prev) => [...prev, { file_id: data.file_id, type: 'local' }])
    },
    onError: (error) => {
      console.error('🎵 uploadAudioMutation onError', error)
      toast.error('Failed to upload audio', {
        description: <div>{error.toString()}</div>,
      })
    },
  })

  const handleFileUpload = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files
      if (files) {
        for (const file of files) {
          const type = file.type.toLowerCase()
          if (type.startsWith('image/')) {
            uploadImageMutation(file)
          } else if (type.startsWith('video/')) {
            uploadVideoMutation(file)
          } else if (type.startsWith('audio/')) {
            uploadAudioMutation(file)
          } else {
            const ext = file.name.split('.').pop()?.toLowerCase() || ''
            if (['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp'].includes(ext)) {
              uploadImageMutation(file)
            } else if (['mp4', 'mov', 'avi', 'mkv', 'webm', 'm4v', '3gp'].includes(ext)) {
              uploadVideoMutation(file)
            } else if (['mp3', 'wav', 'aac', 'm4a', 'ogg', 'flac', 'opus'].includes(ext)) {
              uploadAudioMutation(file)
            } else {
              toast.error('Unknown file type. Please upload image, video, or audio.')
            }
          }
        }
      }
      if (e.target) {
        e.target.value = ''
      }
    },
    [uploadImageMutation, uploadVideoMutation, uploadAudioMutation]
  )

  const handleCancelChat = useCallback(async () => {
    if (sessionId) {
      // 同时取消普通聊天和魔法生成任务
      await Promise.all([cancelChat(sessionId), cancelMagicGenerate(sessionId)])
    }
    onCancelChat?.()
  }, [sessionId, onCancelChat])

  // Send Prompt
  const handleSendPrompt = useCallback(async () => {
    if (pending) return

    // 检查是否使用 Jaaz 服务（任一已选工具来自 jaaz）
    const isUsingJaaz = selectedTools?.some((tool) => tool.provider === 'jaaz')

    // 只有当使用 Jaaz 服务且余额为 0 时才提醒充值
    if (authStatus.is_logged_in && isUsingJaaz && parseFloat(balance) <= 0) {
      toast.error(t('chat:insufficientBalance'), {
        description: <RechargeContent />,
        duration: 10000,
      })
      return
    }

    // 至少选择一个工具（text / image / video 均可）
    if (!selectedTools || selectedTools.length === 0) {
      toast.error(t('chat:textarea.selectModelOrTool', 'Please select at least one model or tool'))
      return
    }

    let text_content: MessageContent[] | string = prompt
    if (prompt.length === 0 || prompt.trim() === '') {
      toast.error(t('chat:textarea.enterPrompt'))
      return
    }

    // Add aspect ratio, quantity and duration information if not default values
    let additionalInfo = ''
    if (selectedAspectRatio !== 'auto') {
      additionalInfo += `<aspect_ratio>${selectedAspectRatio}</aspect_ratio>\n`
    }
    if (quantity !== 1) {
      additionalInfo += `<quantity>${quantity}</quantity>\n`
    }
    if (duration !== 5) {
      additionalInfo += `<duration>${duration}</duration>\n`
    }

    if (additionalInfo) {
      text_content = text_content + '\n\n' + additionalInfo
    }

    if (images.length > 0) {
      text_content += `\n\n<input_images count="${images.length}">`
      images.forEach((image, index) => {
        text_content += `\n<image index="${index + 1}" file_id="${image.file_id}" type="${image.type}" width="${image.width}" height="${image.height}" />`
      })
      text_content += `\n</input_images>`
    }

    if (videos.length > 0) {
      text_content += `\n\n<input_videos count="${videos.length}">`
      videos.forEach((video, index) => {
        text_content += `\n<video index="${index + 1}" file_id="${video.file_id}" type="${video.type}" />`
      })
      text_content += `\n</input_videos>`
    }

    if (audios.length > 0) {
      text_content += `\n\n<input_audios count="${audios.length}">`
      audios.forEach((audio, index) => {
        text_content += `\n<audio index="${index + 1}" file_id="${audio.file_id}" type="${audio.type}" />`
      })
      text_content += `\n</input_audios>`
    }

    // Fetch images as base64; skip any that fail to load
    // - type='local'  → /api/file/{file_id}
    // - type='asset'  → serve_path (/api/material/serve/{disk_name})
    const imagePromises = images.map(async (image) => {
      try {
        const url = image.type === 'asset' && image.serve_path
          ? image.serve_path
          : `/api/file/${image.file_id}`
        const response = await fetch(url)
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const blob = await response.blob()
        return await new Promise<string>((resolve, reject) => {
          const reader = new FileReader()
          reader.onloadend = () => resolve(reader.result as string)
          reader.onerror = () => reject(reader.error)
          reader.readAsDataURL(blob)
        })
      } catch (err) {
        console.error(`Failed to load image ${image.file_id}:`, err)
        return null
      }
    })

    const base64Results = await Promise.all(imagePromises)
    const base64Images = base64Results.filter((b): b is string => b !== null)

    const final_content = [
      {
        type: 'text',
        text: text_content as string,
      },
      ...base64Images.map((url) => ({
        type: 'image_url',
        image_url: { url },
      })),
    ] as MessageContent[]

    const newMessage = messages.concat([
      {
        role: 'user',
        content: final_content,
      },
    ])

    setImages([])
    setVideos([])
    setAudios([])
    setPrompt('')

    onSendMessages(newMessage, {
      toolList: selectedTools && selectedTools.length > 0 ? selectedTools : [],
    })
  }, [
    pending,
    selectedTools,
    prompt,
    onSendMessages,
    images,
    videos,
    audios,
    messages,
    t,
    selectedAspectRatio,
    quantity,
    duration,
    authStatus.is_logged_in,
    setShowLoginDialog,
    balance,
    RechargeContent,
  ])

  // Add an asset from the material library to the chat
  const addAssetToChat = useCallback(async (rec: AssetFileRecord) => {
    if (!rec.disk_name || !rec.serve_url) {
      toast.error('Asset file not available on disk')
      return
    }
    const servePath = rec.serve_url
    const ftype = rec.file_type ?? 'image'
    if (ftype === 'image') {
      // Try to get image dimensions
      let width = 0
      let height = 0
      try {
        await new Promise<void>((resolve) => {
          const img = new window.Image()
          img.onload = () => { width = img.naturalWidth; height = img.naturalHeight; resolve() }
          img.onerror = () => resolve()
          img.src = servePath
        })
      } catch { /* ignore */ }
      setImages((prev) => [
        ...prev,
        { file_id: rec.asset_id, width, height, type: 'asset', serve_path: servePath },
      ])
    } else if (ftype === 'video') {
      setVideos((prev) => [
        ...prev,
        { file_id: rec.asset_id, type: 'asset', serve_path: servePath },
      ])
    } else if (ftype === 'audio') {
      setAudios((prev) => [
        ...prev,
        { file_id: rec.asset_id, type: 'asset', serve_path: servePath },
      ])
    }
    textareaRef.current?.focus()
  }, [])

  // Drop Area
  const dropAreaRef = useRef<HTMLDivElement>(null)
  const [isDragOver, setIsDragOver] = useState(false)

  const handleFilesDrop = useCallback(
    (files: File[]) => {
      for (const file of files) {
        const type = file.type.toLowerCase()
        if (type.startsWith('image/')) {
          uploadImageMutation(file)
        } else if (type.startsWith('video/')) {
          uploadVideoMutation(file)
        } else if (type.startsWith('audio/')) {
          uploadAudioMutation(file)
        } else {
          const ext = file.name.split('.').pop()?.toLowerCase() || ''
          if (['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp'].includes(ext)) {
            uploadImageMutation(file)
          } else if (['mp4', 'mov', 'avi', 'mkv', 'webm', 'm4v', '3gp'].includes(ext)) {
            uploadVideoMutation(file)
          } else if (['mp3', 'wav', 'aac', 'm4a', 'ogg', 'flac', 'opus'].includes(ext)) {
            uploadAudioMutation(file)
          } else {
            toast.error('Unknown file type. Please upload image, video, or audio.')
          }
        }
      }
    },
    [uploadImageMutation, uploadVideoMutation, uploadAudioMutation]
  )

  useDrop(dropAreaRef, {
    onDragOver() {
      setIsDragOver(true)
    },
    onDragLeave() {
      setIsDragOver(false)
    },
    onDrop() {
      setIsDragOver(false)
    },
    onFiles: handleFilesDrop,
  })

  useEffect(() => {
    const handleAddImagesToChat = (data: TCanvasAddImagesToChatEvent) => {
      data.forEach(async (image) => {
        if (image.base64) {
          const file = dataURLToFile(image.base64, image.fileId)
          uploadImageMutation(file)
        } else {
          setImages(
            produce((prev) => {
              prev.push({
                file_id: image.fileId,
                width: image.width,
                height: image.height,
                type: 'local',
              })
            })
          )
        }
      })

      textareaRef.current?.focus()
    }

    const handleMaterialAddImagesToChat = async (
      data: TMaterialAddImagesToChatEvent
    ) => {
      // Legacy event from MaterialManager — re-upload as local file
      data.forEach(async (image: TMaterialAddImagesToChatEvent[0]) => {
        try {
          const fileUrl = `/api/serve_file?file_path=${encodeURIComponent(image.filePath)}`
          const response = await fetch(fileUrl)
          const blob = await response.blob()
          const file = new File([blob], image.fileName, {
            type: `image/${image.fileType}`,
          })
          uploadImageMutation(file)
        } catch (error) {
          console.error('Failed to load image from material:', error)
          toast.error('Failed to load image from material', {
            description: `${error}`,
          })
        }
      })

      textareaRef.current?.focus()
    }

    eventBus.on('Canvas::AddImagesToChat', handleAddImagesToChat)
    eventBus.on('Material::AddImagesToChat', handleMaterialAddImagesToChat)
    return () => {
      eventBus.off('Canvas::AddImagesToChat', handleAddImagesToChat)
      eventBus.off('Material::AddImagesToChat', handleMaterialAddImagesToChat)
    }
  }, [uploadImageMutation])

  // Close quantity slider when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        quantitySliderRef.current &&
        !quantitySliderRef.current.contains(event.target as Node)
      ) {
        setShowQuantitySlider(false)
      }
    }

    if (showQuantitySlider) {
      document.addEventListener('mousedown', handleClickOutside)
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [showQuantitySlider])

  // Close duration slider when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        durationSliderRef.current &&
        !durationSliderRef.current.contains(event.target as Node)
      ) {
        setShowDurationSlider(false)
      }
    }

    if (showDurationSlider) {
      document.addEventListener('mousedown', handleClickOutside)
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [showDurationSlider])

  return (
    <motion.div
      ref={dropAreaRef}
      className={cn(
        'w-full flex flex-col items-center border border-primary/20 rounded-2xl p-3 hover:border-primary/40 transition-all duration-300 cursor-text gap-5 bg-background/80 backdrop-blur-xl relative',
        isFocused && 'border-primary/40',
        className
      )}
      style={{
        boxShadow: isFocused
          ? '0 0 0 4px color-mix(in oklab, var(--primary) 10%, transparent)'
          : 'none',
      }}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3, ease: 'linear' }}
      onClick={() => textareaRef.current?.focus()}
    >
      <AnimatePresence>
        {isDragOver && (
          <motion.div
            className="absolute top-0 left-0 right-0 bottom-0 bg-background/50 backdrop-blur-xl rounded-2xl z-10"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2, ease: 'easeInOut' }}
          >
            <div className="flex items-center justify-center h-full">
              <p className="text-sm text-muted-foreground">
                Drop images, videos, or audio here to upload
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {images.length > 0 && (
          <motion.div
            className="flex items-center gap-2 w-full"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2, ease: 'easeInOut' }}
          >
            {images.map((image) => (
              <motion.div
                key={image.file_id}
                className="relative size-10"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ duration: 0.2, ease: 'easeInOut' }}
              >
                <img
                  key={image.file_id}
                  src={
                    image.type === 'asset' && image.serve_path
                      ? image.serve_path
                      : `/api/file/${image.file_id}`
                  }
                  alt="Uploaded image"
                  className="w-full h-full object-cover rounded-md"
                  draggable={false}
                />
                <Button
                  variant="secondary"
                  size="icon"
                  className="absolute -top-1 -right-1 size-4"
                  onClick={() =>
                    setImages((prev) =>
                      prev.filter((i) => i.file_id !== image.file_id)
                    )
                  }
                >
                  <XIcon className="size-3" />
                </Button>
              </motion.div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {videos.length > 0 && (
          <motion.div
            className="flex items-center gap-2 w-full"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2, ease: 'easeInOut' }}
          >
            {videos.map((video) => (
              <motion.div
                key={video.file_id}
                className="relative size-10"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ duration: 0.2, ease: 'easeInOut' }}
              >
                <div className="w-full h-full bg-muted rounded-md flex items-center justify-center">
                  <Video className="size-5 text-muted-foreground" />
                </div>
                <Button
                  variant="secondary"
                  size="icon"
                  className="absolute -top-1 -right-1 size-4"
                  onClick={() =>
                    setVideos((prev) =>
                      prev.filter((v) => v.file_id !== video.file_id)
                    )
                  }
                >
                  <XIcon className="size-3" />
                </Button>
              </motion.div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {audios.length > 0 && (
          <motion.div
            className="flex items-center gap-2 w-full"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2, ease: 'easeInOut' }}
          >
            {audios.map((audio) => (
              <motion.div
                key={audio.file_id}
                className="relative size-10"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ duration: 0.2, ease: 'easeInOut' }}
              >
                <div className="w-full h-full bg-muted rounded-md flex items-center justify-center">
                  <Music className="size-5 text-muted-foreground" />
                </div>
                <Button
                  variant="secondary"
                  size="icon"
                  className="absolute -top-1 -right-1 size-4"
                  onClick={() =>
                    setAudios((prev) =>
                      prev.filter((a) => a.file_id !== audio.file_id)
                    )
                  }
                >
                  <XIcon className="size-3" />
                </Button>
              </motion.div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      <Textarea
        ref={textareaRef}
        className="w-full h-full border-none outline-none resize-none"
        placeholder={t('chat:textarea.placeholder')}
        value={prompt}
        autoSize
        onChange={(e) => setPrompt(e.target.value)}
        onFocus={() => setIsFocused(true)}
        onBlur={() => setIsFocused(false)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSendPrompt()
          }
        }}
      />

      <div className="flex items-center justify-between gap-2 w-full">
        <div className="flex items-center gap-2 max-w-[calc(100%-50px)] flex-wrap">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,video/*,audio/*"
            multiple
            onChange={handleFileUpload}
            hidden
          />
          <Button
            variant="outline"
            size="sm"
            onClick={() => fileInputRef.current?.click()}
          >
            <PlusIcon className="size-4" />
          </Button>

          {/* Material library asset picker */}
          <Popover
            open={assetPickerOpen}
            onOpenChange={(open) => {
              setAssetPickerOpen(open)
              if (!open) setAssetSearchQuery('')
            }}
          >
            <PopoverTrigger asChild>
              <Button variant="outline" size="sm">
                <Library className="size-4" />
              </Button>
            </PopoverTrigger>
            <PopoverContent align="start" className="flex flex-col p-0 w-[24rem] h-[22rem] min-w-[18rem] min-h-[14rem] resize overflow-hidden">
              {/* Search input */}
              <div className="flex items-center gap-2 px-3 py-2 border-b shrink-0">
                <Search className="size-4 shrink-0 text-muted-foreground" />
                <Input
                  value={assetSearchQuery}
                  onChange={(e) => setAssetSearchQuery(e.target.value)}
                  placeholder="搜索素材…"
                  className="h-7 border-0 p-0 text-sm shadow-none focus-visible:ring-0"
                />
              </div>
              {/* Grid — scrollable */}
              <div className="overflow-y-auto flex-1 p-2">
                {(() => {
                  const filtered = materialFiles.filter(
                    (r) =>
                      r &&
                      r.disk_name &&
                      r.serve_url &&
                      r.status === 'Active' &&
                      (r.name ?? '').toLowerCase().includes(assetSearchQuery.toLowerCase())
                  )
                  if (filtered.length === 0) {
                    return (
                      <p className="py-6 text-center text-sm text-muted-foreground">
                        无匹配素材
                      </p>
                    )
                  }
                  return (
                    <div className="grid grid-cols-3 gap-1.5">
                      {filtered.map((rec) => (
                        <button
                          key={rec.asset_id || rec.fid}
                          onClick={() => {
                            addAssetToChat(rec)
                            setAssetPickerOpen(false)
                          }}
                          className="group flex flex-col items-center gap-1 rounded-md p-1 hover:bg-accent focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                        >
                          {rec.file_type === 'image' && rec.serve_url ? (
                            <img
                              src={rec.serve_url}
                              alt={rec.name || 'Material'}
                              className="h-16 w-full rounded object-cover"
                              onError={(e) => {
                                ;(e.target as HTMLImageElement).style.display = 'none'
                              }}
                            />
                          ) : rec.file_type === 'video' ? (
                            <div className="flex h-16 w-full items-center justify-center rounded bg-muted">
                              <Video className="size-6 text-muted-foreground" />
                            </div>
                          ) : (
                            <div className="flex h-16 w-full items-center justify-center rounded bg-muted">
                              <Music className="size-6 text-muted-foreground" />
                            </div>
                          )}
                          <span className="w-full truncate text-center text-xs text-muted-foreground group-hover:text-foreground">
                            {rec.name || `Asset ${rec.asset_id}`}
                          </span>
                        </button>
                      ))}
                    </div>
                  )
                })()}
              </div>
            </PopoverContent>
          </Popover>

          <ModelSelectorV3 />

          {/* Aspect Ratio Selector */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="outline"
                className="flex items-center gap-1"
                size={'sm'}
              >
                <RectangleVertical className="size-4" />
                <span className="text-sm">{selectedAspectRatio}</span>
                <ChevronDown className="size-3 opacity-50" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="w-32">
              {['auto', '1:1', '4:3', '3:4', '16:9', '9:16'].map((ratio) => (
                <DropdownMenuItem
                  key={ratio}
                  onClick={() => {
                    setSelectedAspectRatio(ratio)
                    localStorage.setItem('chat_aspect_ratio', ratio)
                  }}
                  className="flex items-center justify-between"
                >
                  <span>{ratio}</span>
                  {selectedAspectRatio === ratio && (
                    <div className="size-2 rounded-full bg-primary" />
                  )}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Quantity Selector */}
          <div className="relative" ref={quantitySliderRef}>
            <Button
              variant="outline"
              className="flex items-center gap-1"
              onClick={() => setShowQuantitySlider(!showQuantitySlider)}
              size={'sm'}
            >
              <Hash className="size-4" />
              <span className="text-sm">{quantity}</span>
              <ChevronDown className="size-3 opacity-50" />
            </Button>

            {/* Quantity Slider */}
            <AnimatePresence>
              {showQuantitySlider && (
                <motion.div
                  className="absolute bottom-full mb-2 left-0  bg-background border border-border rounded-lg p-4 shadow-lg min-w-48"
                  initial={{ opacity: 0, y: 10, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 10, scale: 0.95 }}
                  transition={{ duration: 0.15, ease: 'easeOut' }}
                >
                  <div className="flex flex-col gap-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">
                        {t('chat:textarea.quantity', 'Image Quantity')}
                      </span>
                      <span className="text-sm text-muted-foreground">
                        {quantity}
                      </span>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-xs text-muted-foreground">1</span>
                      <input
                        type="range"
                        min="1"
                        max={MAX_QUANTITY}
                        value={quantity}
                        onChange={(e) => {
                          const v = Number(e.target.value)
                          setQuantity(v)
                          localStorage.setItem('chat_quantity', String(v))
                        }}
                        className="flex-1 h-2 bg-muted rounded-lg appearance-none cursor-pointer
                                  [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4
                                  [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary
                                  [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:shadow-sm
                                  [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:rounded-full
                                  [&::-moz-range-thumb]:bg-primary [&::-moz-range-thumb]:cursor-pointer [&::-moz-range-thumb]:border-0"
                      />
                      <span className="text-xs text-muted-foreground">
                        {MAX_QUANTITY}
                      </span>
                    </div>
                  </div>
                  {/* Arrow pointing down */}
                  <div className="absolute top-full left-1/2 -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-border"></div>
                  <div className="absolute top-full left-1/2 -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-background translate-y-[-1px]"></div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Duration Selector */}
          <div className="relative" ref={durationSliderRef}>
            <Button
              variant="outline"
              className="flex items-center gap-1"
              onClick={() => setShowDurationSlider(!showDurationSlider)}
              size={'sm'}
            >
              <Video className="size-4" />
              <span className="text-sm">{duration}s</span>
              <ChevronDown className="size-3 opacity-50" />
            </Button>

            {/* Duration Slider */}
            <AnimatePresence>
              {showDurationSlider && (
                <motion.div
                  className="absolute bottom-full mb-2 left-0 bg-background border border-border rounded-lg p-4 shadow-lg min-w-48"
                  initial={{ opacity: 0, y: 10, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 10, scale: 0.95 }}
                  transition={{ duration: 0.15, ease: 'easeOut' }}
                >
                  <div className="flex flex-col gap-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">
                        {t('chat:textarea.duration', 'Video Duration')}
                      </span>
                      <span className="text-sm text-muted-foreground">
                        {duration}s
                      </span>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-xs text-muted-foreground">{MIN_DURATION}</span>
                      <input
                        type="range"
                        min={MIN_DURATION}
                        max={MAX_DURATION}
                        step={1}
                        value={duration}
                        onChange={(e) => {
                          const v = Number(e.target.value)
                          setDuration(v)
                          localStorage.setItem('chat_duration', String(v))
                        }}
                        className="flex-1 h-2 bg-muted rounded-lg appearance-none cursor-pointer
                                  [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4
                                  [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary
                                  [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:shadow-sm
                                  [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:rounded-full
                                  [&::-moz-range-thumb]:bg-primary [&::-moz-range-thumb]:cursor-pointer [&::-moz-range-thumb]:border-0"
                      />
                      <span className="text-xs text-muted-foreground">
                        {MAX_DURATION}
                      </span>
                    </div>
                  </div>
                  {/* Arrow pointing down */}
                  <div className="absolute top-full left-1/2 -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-border"></div>
                  <div className="absolute top-full left-1/2 -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-background translate-y-[-1px]"></div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {pending ? (
          <Button
            className="shrink-0 relative"
            variant="default"
            size="icon"
            onClick={handleCancelChat}
          >
            <Loader2 className="size-5.5 animate-spin absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
            <Square className="size-2 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
          </Button>
        ) : (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  className="shrink-0"
                  variant="default"
                  size="icon"
                  onClick={handleSendPrompt}
                  disabled={!selectedTools?.length || prompt.length === 0}
                >
                  <ArrowUp className="size-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="top">
                {!selectedTools?.length
                  ? t('chat:textarea.selectModelOrTool', 'Please select at least one model or tool')
                  : prompt.length === 0
                    ? t('chat:textarea.enterPrompt', 'Please enter a message')
                    : t('chat:textarea.send', 'Send message')}
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
      </div>
    </motion.div>
  )
}

export default ChatTextarea
