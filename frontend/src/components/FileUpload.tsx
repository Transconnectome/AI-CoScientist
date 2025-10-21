import { useState, useRef } from 'react'
import { Upload, FileText, AlertCircle, Loader2, CheckCircle } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from './ui/button'

interface FileUploadProps {
  onUpload: (file: File) => void | Promise<void>
  maxSize?: number
  accept?: string[]
}

export function FileUpload({
  onUpload,
  maxSize = 10 * 1024 * 1024, // 10MB default
  accept = ['.pdf', '.docx'],
}: FileUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const acceptTypes = {
    '.pdf': 'application/pdf',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  }

  const validateFile = (file: File): string | null => {
    // Check file type
    const extension = '.' + file.name.split('.').pop()?.toLowerCase()
    if (!accept.includes(extension)) {
      return 'Only PDF and DOCX files are accepted'
    }

    // Check file size
    if (file.size > maxSize) {
      return 'File too large. Maximum size is 10MB'
    }

    return null
  }

  const handleFile = async (file: File) => {
    setError(null)

    const validationError = validateFile(file)
    if (validationError) {
      setError(validationError)
      return
    }

    setIsUploading(true)
    setUploadedFile(file)

    try {
      await onUpload(file)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
      setUploadedFile(null)
    } finally {
      setIsUploading(false)
    }
  }

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      handleFile(file)
    }
  }

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setIsDragOver(false)
  }

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setIsDragOver(false)

    const file = event.dataTransfer.files?.[0]
    if (file) {
      handleFile(file)
    }
  }

  const handleClick = () => {
    inputRef.current?.click()
  }

  const handleRetry = () => {
    setError(null)
    setUploadedFile(null)
    if (inputRef.current) {
      inputRef.current.value = ''
    }
  }

  return (
    <div className="w-full max-w-xl">
      <div
        data-testid="drop-zone"
        className={cn(
          'relative rounded-lg border-2 border-dashed transition-colors p-8',
          isDragOver && 'border-primary bg-primary/5',
          !isDragOver && 'border-border hover:border-primary/50',
          isUploading && 'pointer-events-none opacity-60'
        )}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
      >
        <input
          ref={inputRef}
          type="file"
          accept={Object.values(acceptTypes).join(',')}
          onChange={handleFileChange}
          disabled={isUploading}
          className="sr-only"
          id="file-upload"
          aria-label="Upload your paper (PDF or DOCX)"
        />

        <div className="flex flex-col items-center justify-center gap-4 text-center">
          {isUploading ? (
            <>
              <Loader2 className="h-12 w-12 animate-spin text-primary" />
              <div>
                <p className="text-lg font-medium">Uploading...</p>
                <p className="text-sm text-muted-foreground">{uploadedFile?.name}</p>
              </div>
            </>
          ) : uploadedFile && !error ? (
            <>
              <CheckCircle className="h-12 w-12 text-green-600" />
              <div>
                <p className="text-lg font-medium">File uploaded successfully</p>
                <p className="text-sm text-muted-foreground">{uploadedFile.name}</p>
              </div>
            </>
          ) : (
            <>
              <Upload className="h-12 w-12 text-muted-foreground" />
              <div>
                <p className="text-lg font-medium">
                  Drag and drop your paper here
                </p>
                <p className="text-sm text-muted-foreground">
                  or click to browse
                </p>
              </div>
              <p className="text-xs text-muted-foreground">
                Supports PDF and DOCX files up to 10MB
              </p>
            </>
          )}
        </div>
      </div>

      {error && (
        <div className="mt-4 rounded-md bg-destructive/10 p-4">
          <div className="flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-destructive flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="text-sm font-medium text-destructive">{error}</p>
              <Button
                variant="outline"
                size="sm"
                onClick={handleRetry}
                className="mt-2"
              >
                Try Again
              </Button>
            </div>
          </div>
        </div>
      )}

      {uploadedFile && !error && (
        <div className="mt-4 rounded-md bg-muted p-4">
          <div className="flex items-center gap-3">
            <FileText className="h-5 w-5 text-primary" />
            <div className="flex-1">
              <p className="text-sm font-medium">{uploadedFile.name}</p>
              <p className="text-xs text-muted-foreground">
                {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
