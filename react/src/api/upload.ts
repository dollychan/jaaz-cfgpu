import { compressImageFile } from '@/utils/imageUtils'

export async function uploadImage(
  file: File
): Promise<{ file_id: string; width: number; height: number; url: string }> {
  // Compress image before upload
  const compressedFile = await compressImageFile(file)

  const formData = new FormData()
  formData.append('file', compressedFile)
  const response = await fetch('/api/upload_image', {
    method: 'POST',
    body: formData,
  })
  return await response.json()
}

export async function uploadVideo(
  file: File
): Promise<{ file_id: string; url: string }> {
  const formData = new FormData()
  formData.append('file', file)
  const response = await fetch('/api/upload_video', {
    method: 'POST',
    body: formData,
  })
  return await response.json()
}
