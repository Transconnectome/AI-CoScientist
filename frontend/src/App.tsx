import { Routes, Route } from 'react-router-dom'
import { Dashboard } from './pages/Dashboard'
import { Upload } from './pages/Upload'
import { FileUploadDemo } from './pages/FileUploadDemo'
import { PaperDetails } from './pages/PaperDetails'
import { NotFound } from './pages/NotFound'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Dashboard />} />
      <Route path="/upload" element={<Upload />} />
      <Route path="/demo/file-upload" element={<FileUploadDemo />} />
      <Route path="/papers/:paperId" element={<PaperDetails />} />
      <Route path="*" element={<NotFound />} />
    </Routes>
  )
}

export default App
