import { Input } from '@/components/ui/input'
import CanvasExport from './CanvasExport'
import TopMenu from '../TopMenu'
import { useState, useEffect } from 'react'

type CanvasHeaderProps = {
  initialName: string
  canvasId: string
  onNameSave: (name: string) => void
}

const CanvasHeader: React.FC<CanvasHeaderProps> = ({
  initialName,
  canvasId,
  onNameSave,
}) => {
  // 内部维护 name state，键入时只有 CanvasHeader 自己重渲，
  // 不会触发父级 Canvas（及其子树 Excalidraw）的 re-render。
  const [name, setName] = useState(initialName)

  // 当外部 initialName 加载完成后同步（首次从 API 取到数据）
  useEffect(() => {
    setName(initialName)
  }, [initialName])

  return (
    <TopMenu
      middle={
        <Input
          className="text-sm text-muted-foreground text-center bg-transparent border-none shadow-none w-fit h-7 hover:bg-primary-foreground transition-all"
          value={name}
          onChange={(e) => setName(e.target.value)}
          onBlur={() => onNameSave(name)}
        />
      }
      right={<CanvasExport />}
    />
  )
}

export default CanvasHeader
