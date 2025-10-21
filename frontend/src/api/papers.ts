import { apiClient } from './client'
import type { PaperData } from '@/types/paper'

export interface Paper {
  id: string
  title: string
  status: 'uploaded' | 'analyzing' | 'complete'
  overall_score?: number
}

export const papersApi = {
  upload: async (file: File) => {
    const formData = new FormData()
    formData.append('file', file)

    const response = await apiClient.post('/api/v1/papers/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  getById: async (id: string): Promise<Paper> => {
    const response = await apiClient.get(`/api/v1/papers/${id}`)
    return response.data
  },

  getPaperById: async (id: string): Promise<PaperData> => {
    const response = await apiClient.get(`/api/v1/papers/${id}/details`)
    return response.data
  },
}
