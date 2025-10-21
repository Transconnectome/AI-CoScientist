export interface PaperMetadata {
  id: string
  title: string
  authors: string[]
  abstract: string
  uploadedAt: string
  status: 'processing' | 'complete' | 'error'
  fileSize?: number
  fileName?: string
}

export interface CategoryScore {
  category: string
  score: number
  confidence: number
  explanation?: string
}

export interface PaperScores {
  overall: number
  novelty: CategoryScore
  technical_quality: CategoryScore
  clarity: CategoryScore
  impact: CategoryScore
}

export interface Recommendation {
  id: string
  category: 'novelty' | 'technical_quality' | 'clarity' | 'impact' | 'general'
  priority: 'high' | 'medium' | 'low'
  title: string
  description: string
  impact: string
  timestamp?: string
}

export interface AnalysisStage {
  stage: string
  status: 'pending' | 'in_progress' | 'complete' | 'error'
  timestamp?: string
  duration?: number
  description?: string
}

export interface PaperData {
  metadata: PaperMetadata
  scores?: PaperScores
  recommendations?: Recommendation[]
  stages?: AnalysisStage[]
}

export type PaperStatus = 'processing' | 'complete' | 'error'
