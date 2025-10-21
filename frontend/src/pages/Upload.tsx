import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { FileUpload } from '@/components/FileUpload'
import { papersApi } from '@/api/papers'
import { AlertCircle, CheckCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'

export function Upload() {
  const navigate = useNavigate()
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle')
  const [uploadedPaperId, setUploadedPaperId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [uploadProgress, setUploadProgress] = useState<number>(0)

  const handleUpload = async (file: File) => {
    setUploadStatus('uploading')
    setError(null)
    setUploadProgress(0)

    try {
      // Simulate progress updates (in real implementation, this would track actual upload)
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return 90
          }
          return prev + 10
        })
      }, 100)

      const response = await papersApi.upload(file)

      clearInterval(progressInterval)
      setUploadProgress(100)
      setUploadedPaperId(response.paper_id)
      setUploadStatus('success')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
      setUploadStatus('error')
      setUploadProgress(0)
      throw err // Re-throw so FileUpload can handle it
    }
  }

  const handleViewPaper = () => {
    if (uploadedPaperId) {
      navigate(`/papers/${uploadedPaperId}`)
    }
  }

  return (
    <div className="container mx-auto p-8 max-w-4xl">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Upload Paper</h1>
        <p className="text-lg text-muted-foreground">
          Upload your research paper for AI-powered analysis and improvement suggestions
        </p>
      </div>

      <FileUpload onUpload={handleUpload} />

      {uploadStatus === 'uploading' && uploadProgress > 0 && (
        <div className="mt-4 rounded-lg bg-blue-50 dark:bg-blue-950 p-4">
          <div className="flex items-center gap-3 mb-2">
            <p className="text-sm font-medium text-blue-900 dark:text-blue-100">
              Uploading... {uploadProgress}%
            </p>
          </div>
          <div className="w-full bg-blue-200 dark:bg-blue-900 rounded-full h-2">
            <div
              className="bg-blue-600 dark:bg-blue-400 h-2 rounded-full transition-all duration-300"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
        </div>
      )}

      {uploadStatus === 'success' && uploadedPaperId && (
        <div className="mt-6 rounded-lg bg-green-50 dark:bg-green-950 p-6">
          <div className="flex items-start gap-3">
            <CheckCircle className="h-6 w-6 text-green-600 dark:text-green-400 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-green-900 dark:text-green-100 mb-2">
                Upload Successful!
              </h3>
              <p className="text-sm text-green-800 dark:text-green-200 mb-4">
                Your paper has been uploaded and is ready for analysis.
              </p>
              <div className="flex gap-3">
                <Button onClick={handleViewPaper}>
                  View Paper Analysis
                </Button>
                <Button variant="outline" onClick={() => window.location.reload()}>
                  Upload Another
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {uploadStatus === 'error' && error && (
        <div className="mt-6 rounded-lg bg-red-50 dark:bg-red-950 p-6">
          <div className="flex items-start gap-3">
            <AlertCircle className="h-6 w-6 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-red-900 dark:text-red-100 mb-2">
                Upload Failed
              </h3>
              <p className="text-sm text-red-800 dark:text-red-200">
                {error}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
