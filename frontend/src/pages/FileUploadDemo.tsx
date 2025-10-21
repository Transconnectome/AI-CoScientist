import { FileUpload } from '@/components/FileUpload'
import { Card } from '@/components/ui/card'

export function FileUploadDemo() {
  const handleDemoUpload = async (file: File) => {
    // Simulate upload delay for demo
    await new Promise(resolve => setTimeout(resolve, 2000))
    console.log('Demo upload completed:', file.name)
  }

  const handleFailingUpload = async (file: File) => {
    // Simulate upload delay then fail
    await new Promise(resolve => setTimeout(resolve, 1500))
    throw new Error('Demo error: Simulated upload failure')
  }

  return (
    <div className="container mx-auto p-8 max-w-6xl">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">FileUpload Component Demo</h1>
        <p className="text-lg text-muted-foreground">
          Interactive demonstration of the FileUpload component with different configurations
        </p>
      </div>

      <div className="grid gap-8 md:grid-cols-2">
        {/* Default Configuration */}
        <Card className="p-6">
          <h2 className="text-2xl font-semibold mb-4">Default Configuration</h2>
          <p className="text-sm text-muted-foreground mb-4">
            Standard file upload with PDF and DOCX support, 10MB limit
          </p>
          <FileUpload onUpload={handleDemoUpload} />
        </Card>

        {/* Custom Max Size */}
        <Card className="p-6">
          <h2 className="text-2xl font-semibold mb-4">Custom Max Size (5MB)</h2>
          <p className="text-sm text-muted-foreground mb-4">
            Same as default but with 5MB file size limit
          </p>
          <FileUpload onUpload={handleDemoUpload} maxSize={5 * 1024 * 1024} />
        </Card>

        {/* PDF Only */}
        <Card className="p-6">
          <h2 className="text-2xl font-semibold mb-4">PDF Only</h2>
          <p className="text-sm text-muted-foreground mb-4">
            Restricted to PDF files only
          </p>
          <FileUpload onUpload={handleDemoUpload} accept={['.pdf']} />
        </Card>

        {/* Simulated Error */}
        <Card className="p-6">
          <h2 className="text-2xl font-semibold mb-4">Error Handling Demo</h2>
          <p className="text-sm text-muted-foreground mb-4">
            Upload will always fail to demonstrate error handling
          </p>
          <FileUpload onUpload={handleFailingUpload} />
        </Card>
      </div>

      {/* Features Documentation */}
      <Card className="p-6 mt-8">
        <h2 className="text-2xl font-semibold mb-4">Component Features</h2>
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <h3 className="font-semibold mb-2">📁 File Input Methods</h3>
            <ul className="list-disc list-inside text-sm space-y-1 text-muted-foreground">
              <li>Click to browse and select file</li>
              <li>Drag and drop file onto upload zone</li>
              <li>Visual feedback on drag over</li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold mb-2">✅ File Validation</h3>
            <ul className="list-disc list-inside text-sm space-y-1 text-muted-foreground">
              <li>File type validation (PDF, DOCX)</li>
              <li>File size limit enforcement</li>
              <li>Clear error messages</li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold mb-2">⏳ Upload States</h3>
            <ul className="list-disc list-inside text-sm space-y-1 text-muted-foreground">
              <li>Idle state with instructions</li>
              <li>Loading state with spinner</li>
              <li>Success state with confirmation</li>
              <li>Error state with retry option</li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold mb-2">🎨 UI Features</h3>
            <ul className="list-disc list-inside text-sm space-y-1 text-muted-foreground">
              <li>Responsive design</li>
              <li>Dark mode support</li>
              <li>Accessible with ARIA labels</li>
              <li>File info display (name, size)</li>
            </ul>
          </div>
        </div>
      </Card>

      {/* Usage Example */}
      <Card className="p-6 mt-8">
        <h2 className="text-2xl font-semibold mb-4">Usage Example</h2>
        <pre className="bg-muted p-4 rounded-md overflow-x-auto text-sm">
          <code>{`import { FileUpload } from '@/components/FileUpload'

function MyUploadPage() {
  const handleUpload = async (file: File) => {
    // Your upload logic here
    const response = await api.upload(file)
    console.log('Uploaded:', response)
  }

  return (
    <FileUpload
      onUpload={handleUpload}
      maxSize={10 * 1024 * 1024}  // 10MB
      accept={['.pdf', '.docx']}   // Optional
    />
  )
}`}</code>
        </pre>
      </Card>

      {/* Props Documentation */}
      <Card className="p-6 mt-8">
        <h2 className="text-2xl font-semibold mb-4">Props API</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="border-b">
              <tr className="text-left">
                <th className="pb-2 font-semibold">Prop</th>
                <th className="pb-2 font-semibold">Type</th>
                <th className="pb-2 font-semibold">Default</th>
                <th className="pb-2 font-semibold">Description</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              <tr>
                <td className="py-2 font-mono">onUpload</td>
                <td className="py-2 text-muted-foreground">
                  (file: File) =&gt; void | Promise&lt;void&gt;
                </td>
                <td className="py-2 text-muted-foreground">Required</td>
                <td className="py-2 text-muted-foreground">
                  Callback function called when file is selected
                </td>
              </tr>
              <tr>
                <td className="py-2 font-mono">maxSize</td>
                <td className="py-2 text-muted-foreground">number</td>
                <td className="py-2 text-muted-foreground">10485760 (10MB)</td>
                <td className="py-2 text-muted-foreground">
                  Maximum file size in bytes
                </td>
              </tr>
              <tr>
                <td className="py-2 font-mono">accept</td>
                <td className="py-2 text-muted-foreground">string[]</td>
                <td className="py-2 text-muted-foreground">['.pdf', '.docx']</td>
                <td className="py-2 text-muted-foreground">
                  Allowed file extensions
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}
