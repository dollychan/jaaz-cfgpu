export type AssetFileRecord = {
  fid: number
  name: string          // display name (original filename)
  asset_id: string      // e.g. "asset-20260331115141-w2fhz"
  group_id: string
  status: 'Processing' | 'Active' | 'Failed' | string
  asset_type: 'Image' | 'Video' | 'Audio' | string
  project_name: string
  url: string           // CFGPU CDN URL (valid 12h), may be empty while Processing
  created_at: string
  updated_at: string
  // enriched by server
  disk_name: string | null
  file_type: 'image' | 'video' | 'audio' | 'file' | null
  /** Local serve URL, e.g. /api/material/serve/asset-xxx.jpg */
  serve_url: string | null
}

export async function getMaterialFilesApi(): Promise<AssetFileRecord[]> {
  const res = await fetch('/api/material/files')
  if (!res.ok) throw new Error(`Failed to fetch materials: ${res.status}`)
  const data: (Omit<AssetFileRecord, 'serve_url'> & { url?: string })[] = await res.json()
  // map server `url` field (local serve path) to `serve_url` for clarity
  return data.map((r) => ({
    ...r,
    serve_url: (r as any).url ?? null,
  }))
}

export async function pollMaterialStatusesApi(): Promise<{
  updated: string[]
  records: AssetFileRecord[]
}> {
  const res = await fetch('/api/material/poll-status', { method: 'POST' })
  if (!res.ok) throw new Error(`Failed to poll material statuses: ${res.status}`)
  return res.json()
}

export async function renameMaterialApi(assetId: string, name: string): Promise<void> {
  const res = await fetch(`/api/material/rename/${encodeURIComponent(assetId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  })
  if (!res.ok) throw new Error(`Failed to rename material: ${res.status}`)
}

export async function deleteMaterialApi(assetId: string): Promise<void> {
  const res = await fetch(`/api/material/${encodeURIComponent(assetId)}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw new Error(`Failed to delete material: ${res.status}`)
}
