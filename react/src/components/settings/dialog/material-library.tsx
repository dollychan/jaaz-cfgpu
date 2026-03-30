import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Button } from '@/components/ui/button'
import { getConfig, updateConfig } from '@/api/config'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import { Save } from 'lucide-react'

type MaterialLibraryConfig = {
  ak: string
  sk: string
  group_id: string
  project_name: string
  public_base_url: string
}

const DEFAULT: MaterialLibraryConfig = {
  ak: '',
  sk: '',
  group_id: '',
  project_name: 'default',
  public_base_url: '',
}

export default function SettingMaterialLibrary() {
  const [cfg, setCfg] = useState<MaterialLibraryConfig>(DEFAULT)
  const [allConfigs, setAllConfigs] = useState<Record<string, any>>({})
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    getConfig().then((configs: Record<string, any>) => {
      setAllConfigs(configs)
      const ml = configs.material_library || {}
      setCfg({ ...DEFAULT, ...ml })
    })
  }, [])

  const handleChange = (field: keyof MaterialLibraryConfig, value: string) => {
    setCfg((prev) => ({ ...prev, [field]: value }))
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      const result = await updateConfig({ ...allConfigs, material_library: cfg })
      if (result.status === 'success') {
        toast.success('素材资产库配置已保存')
      } else {
        throw new Error(result.message)
      }
    } catch (e: any) {
      toast.error(e.message || '保存失败')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="p-6 space-y-6 max-w-xl">
      <div>
        <h2 className="text-xl font-bold mb-1">素材资产库</h2>
        <p className="text-sm text-gray-500">配置 Volcengine Ark 素材库凭证，上传后自动调用 CreateAsset 接口</p>
      </div>

      <div className="space-y-4">
        <div className="space-y-1">
          <Label>Access Key (AK)</Label>
          <Input
            type="password"
            placeholder="your volcengine AK"
            value={cfg.ak}
            onChange={(e) => handleChange('ak', e.target.value)}
          />
        </div>
        <div className="space-y-1">
          <Label>Secret Key (SK)</Label>
          <Input
            type="password"
            placeholder="your volcengine SK"
            value={cfg.sk}
            onChange={(e) => handleChange('sk', e.target.value)}
          />
        </div>
        <div className="space-y-1">
          <Label>Group ID</Label>
          <Input
            placeholder="group-xxxxxxxxxxxxxxxx"
            value={cfg.group_id}
            onChange={(e) => handleChange('group_id', e.target.value)}
          />
        </div>
        <div className="space-y-1">
          <Label>Project Name（可选，默认 default）</Label>
          <Input
            placeholder="default"
            value={cfg.project_name}
            onChange={(e) => handleChange('project_name', e.target.value)}
          />
        </div>
        <div className="space-y-1">
          <Label>服务器公网访问 Base URL</Label>
          <p className="text-xs text-gray-400">用于生成 CreateAsset 的素材访问 URL，例如 https://yourdomain.com</p>
          <Input
            placeholder="https://yourdomain.com"
            value={cfg.public_base_url}
            onChange={(e) => handleChange('public_base_url', e.target.value)}
          />
        </div>
      </div>

      <Button onClick={handleSave} disabled={saving}>
        <Save className="w-4 h-4 mr-2" />
        {saving ? '保存中...' : '保存'}
      </Button>
    </div>
  )
}
