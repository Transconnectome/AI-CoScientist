import { test, expect } from '@playwright/test'

test.describe('Upload Page - Smoke Tests', () => {
  test('should display upload page with all required elements', async ({ page }) => {
    await page.goto('/upload')

    // Check page title
    await expect(page.getByRole('heading', { name: /upload paper/i })).toBeVisible()

    // Check description
    await expect(page.getByText(/AI-powered analysis/i)).toBeVisible()

    // Check FileUpload component is present
    await expect(page.getByLabelText(/upload.*paper/i)).toBeVisible()

    // Check drag-drop instructions
    await expect(page.getByText(/drag and drop/i)).toBeVisible()
    await expect(page.getByText(/or.*click/i)).toBeVisible()

    // Check file type support message
    await expect(page.getByText(/pdf.*docx/i)).toBeVisible()
  })

  test('should display file info after selecting a file', async ({ page }) => {
    await page.goto('/upload')

    // Create a mock file to upload
    const buffer = Buffer.from('%PDF-1.4 Test PDF Content')
    await page.setInputFiles('input[type="file"]', {
      name: 'test-paper.pdf',
      mimeType: 'application/pdf',
      buffer,
    })

    // Check uploading state appears
    await expect(page.getByText(/uploading/i)).toBeVisible()

    // Note: Without actual backend, upload will fail
    // This test just verifies the UI responds to file selection
  })
})

test.describe('FileUpload Demo Page - Smoke Tests', () => {
  test('should display demo page with all sections', async ({ page }) => {
    await page.goto('/demo/file-upload')

    // Check page title
    await expect(page.getByRole('heading', { name: /fileupload component demo/i, level: 1 })).toBeVisible()

    // Check description
    await expect(page.getByText(/interactive demonstration/i)).toBeVisible()

    // Check all demo card sections are visible
    await expect(page.getByText(/default configuration/i)).toBeVisible()
    await expect(page.getByText(/custom max size.*5mb/i)).toBeVisible()
    await expect(page.getByText(/pdf only/i)).toBeVisible()
    await expect(page.getByText(/error handling demo/i)).toBeVisible()

    // Check features documentation section
    await expect(page.getByRole('heading', { name: /component features/i })).toBeVisible()
    await expect(page.getByText(/file input methods/i)).toBeVisible()
    await expect(page.getByText(/file validation/i)).toBeVisible()
    await expect(page.getByText(/upload states/i)).toBeVisible()
    await expect(page.getByText(/ui features/i)).toBeVisible()

    // Check usage example section
    await expect(page.getByRole('heading', { name: /usage example/i })).toBeVisible()
    await expect(page.locator('pre code')).toContainText('FileUpload')

    // Check props API documentation
    await expect(page.getByRole('heading', { name: /props api/i })).toBeVisible()
    await expect(page.getByRole('table')).toBeVisible()
    await expect(page.getByText(/onUpload/)).toBeVisible()
    await expect(page.getByText(/maxSize/)).toBeVisible()
    await expect(page.getByText(/accept/)).toBeVisible()
  })

  test('should have multiple upload components on demo page', async ({ page }) => {
    await page.goto('/demo/file-upload')

    // Count file upload inputs (should be 4 variations)
    const uploadInputs = page.getByLabelText(/upload.*paper/i)
    await expect(uploadInputs).toHaveCount(4)
  })

  test('should demonstrate different file upload variations', async ({ page }) => {
    await page.goto('/demo/file-upload')

    // Create a small PDF buffer
    const pdfBuffer = Buffer.from('%PDF-1.4 Test PDF Content')

    // Test default configuration (first input)
    const defaultInput = page.getByLabelText(/upload.*paper/i).first()
    await defaultInput.setInputFiles({
      name: 'test.pdf',
      mimeType: 'application/pdf',
      buffer: pdfBuffer,
    })

    // Check uploading state appears
    await expect(page.getByText(/uploading/i).first()).toBeVisible({ timeout: 1000 })

    // After demo upload completes (2s delay)
    await expect(page.getByText(/file uploaded successfully/i).first()).toBeVisible({ timeout: 3000 })
  })

  test('should demonstrate error handling', async ({ page }) => {
    await page.goto('/demo/file-upload')

    const pdfBuffer = Buffer.from('%PDF-1.4 Test PDF Content')

    // Test error demo (4th input)
    const errorInput = page.getByLabelText(/upload.*paper/i).nth(3)
    await errorInput.setInputFiles({
      name: 'test.pdf',
      mimeType: 'application/pdf',
      buffer: pdfBuffer,
    })

    // Should show uploading
    await expect(page.getByText(/uploading/i)).toBeVisible({ timeout: 1000 })

    // Should show error after failed upload (1.5s delay)
    await expect(page.getByText(/demo error/i)).toBeVisible({ timeout: 3000 })

    // Try Again button should appear
    await expect(page.getByRole('button', { name: /try again/i })).toBeVisible()
  })
})

test.describe('Dashboard - Smoke Test', () => {
  test('should display dashboard page', async ({ page }) => {
    await page.goto('/')

    // Check dashboard header
    await expect(page.getByRole('heading', { name: /paper analysis dashboard/i })).toBeVisible()

    // Check welcome message
    await expect(page.getByText(/welcome to AI-CoScientist/i)).toBeVisible()
  })
})
