import { test, expect } from '@playwright/test'
import path from 'path'

test.describe('Upload Page E2E', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/upload')
  })

  test('should display upload page with title and FileUpload component', async ({ page }) => {
    // Check page title
    await expect(page.getByRole('heading', { name: /upload paper/i })).toBeVisible()

    // Check description
    await expect(page.getByText(/AI-powered analysis/i)).toBeVisible()

    // Check FileUpload component is present
    await expect(page.getByLabelText(/upload.*paper/i)).toBeVisible()

    // Check drag-drop instructions
    await expect(page.getByText(/drag and drop/i)).toBeVisible()
  })

  test('should upload PDF file successfully (mocked)', async ({ page }) => {
    // Mock the API endpoint
    await page.route('**/api/papers/upload', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          paper_id: 'test-paper-123',
          status: 'uploaded',
        }),
      })
    })

    // Create a test file
    const testFile = path.join(__dirname, '../test-fixtures/test-paper.pdf')

    // Upload the file
    const fileInput = page.getByLabelText(/upload.*paper/i)
    await fileInput.setInputFiles(testFile)

    // Wait for upload to complete (check for success message)
    await expect(page.getByText(/upload successful/i)).toBeVisible({ timeout: 5000 })

    // Check View Paper Analysis button appears
    await expect(page.getByRole('button', { name: /view paper analysis/i })).toBeVisible()

    // Check Upload Another button appears
    await expect(page.getByRole('button', { name: /upload another/i })).toBeVisible()
  })

  test('should show progress bar during upload', async ({ page }) => {
    // Mock slow API endpoint
    await page.route('**/api/papers/upload', async (route) => {
      // Simulate slow upload
      await new Promise(resolve => setTimeout(resolve, 2000))
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          paper_id: 'test-paper-123',
          status: 'uploaded',
        }),
      })
    })

    const testFile = path.join(__dirname, '../test-fixtures/test-paper.pdf')
    const fileInput = page.getByLabelText(/upload.*paper/i)
    await fileInput.setInputFiles(testFile)

    // Check progress indicator appears
    await expect(page.getByText(/uploading/i)).toBeVisible()

    // Wait for completion
    await expect(page.getByText(/upload successful/i)).toBeVisible({ timeout: 5000 })
  })

  test('should handle upload error gracefully', async ({ page }) => {
    // Mock failing API endpoint
    await page.route('**/api/papers/upload', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({
          error: 'Server error: Failed to process upload',
        }),
      })
    })

    const testFile = path.join(__dirname, '../test-fixtures/test-paper.pdf')
    const fileInput = page.getByLabelText(/upload.*paper/i)
    await fileInput.setInputFiles(testFile)

    // Check error message appears
    await expect(page.getByText(/upload failed/i)).toBeVisible({ timeout: 5000 })

    // Check success message doesn't appear
    await expect(page.getByText(/upload successful/i)).not.toBeVisible()
  })

  test('should navigate to paper details when clicking View Paper Analysis', async ({ page }) => {
    // Mock the API endpoint
    await page.route('**/api/papers/upload', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          paper_id: 'test-paper-456',
          status: 'uploaded',
        }),
      })
    })

    const testFile = path.join(__dirname, '../test-fixtures/test-paper.pdf')
    const fileInput = page.getByLabelText(/upload.*paper/i)
    await fileInput.setInputFiles(testFile)

    // Wait for success
    await expect(page.getByRole('button', { name: /view paper analysis/i })).toBeVisible({ timeout: 5000 })

    // Click View Paper Analysis
    await page.getByRole('button', { name: /view paper analysis/i }).click()

    // Check navigation occurred
    await expect(page).toHaveURL(/\/papers\/test-paper-456/)
  })

  test('should show file validation error for invalid file type', async ({ page }) => {
    // Create an invalid file (image instead of PDF/DOCX)
    const invalidFile = path.join(__dirname, '../test-fixtures/test-image.jpg')

    // Try to upload invalid file
    const fileInput = page.getByLabelText(/upload.*paper/i)
    await fileInput.setInputFiles(invalidFile)

    // Check error message appears
    await expect(page.getByText(/only.*pdf.*docx/i)).toBeVisible()

    // Check upload didn't proceed
    await expect(page.getByText(/upload successful/i)).not.toBeVisible()
  })

  test('should allow retry after upload failure', async ({ page }) => {
    let attemptCount = 0

    // Mock API to fail first time, succeed second time
    await page.route('**/api/papers/upload', async (route) => {
      attemptCount++
      if (attemptCount === 1) {
        await route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'First attempt failed' }),
        })
      } else {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            paper_id: 'test-paper-789',
            status: 'uploaded',
          }),
        })
      }
    })

    const testFile = path.join(__dirname, '../test-fixtures/test-paper.pdf')

    // First attempt - should fail
    const fileInput = page.getByLabelText(/upload.*paper/i)
    await fileInput.setInputFiles(testFile)
    await expect(page.getByText(/upload failed/i)).toBeVisible({ timeout: 5000 })

    // Click Try Again button
    await page.getByRole('button', { name: /try again/i }).click()

    // Second attempt - should succeed
    await fileInput.setInputFiles(testFile)
    await expect(page.getByText(/upload successful/i)).toBeVisible({ timeout: 5000 })
  })
})

test.describe('FileUpload Demo Page E2E', () => {
  test('should display demo page with all component variations', async ({ page }) => {
    await page.goto('/demo/file-upload')

    // Check page title
    await expect(page.getByRole('heading', { name: /fileupload component demo/i })).toBeVisible()

    // Check all demo sections are visible
    await expect(page.getByText(/default configuration/i)).toBeVisible()
    await expect(page.getByText(/custom max size/i)).toBeVisible()
    await expect(page.getByText(/pdf only/i)).toBeVisible()
    await expect(page.getByText(/error handling demo/i)).toBeVisible()

    // Check features documentation
    await expect(page.getByText(/component features/i)).toBeVisible()

    // Check usage example code
    await expect(page.getByText(/usage example/i)).toBeVisible()

    // Check props API documentation
    await expect(page.getByText(/props api/i)).toBeVisible()
  })

  test('should demonstrate file upload in default configuration section', async ({ page }) => {
    await page.goto('/demo/file-upload')

    const testFile = path.join(__dirname, '../test-fixtures/test-paper.pdf')

    // Find the first upload input (default configuration)
    const uploadInputs = page.getByLabelText(/upload.*paper/i)
    const firstInput = uploadInputs.first()

    // Upload file
    await firstInput.setInputFiles(testFile)

    // Check upload completes (simulated 2s delay)
    await expect(page.getByText(/file uploaded successfully/i).first()).toBeVisible({ timeout: 5000 })
  })

  test('should demonstrate error handling in error demo section', async ({ page }) => {
    await page.goto('/demo/file-upload')

    const testFile = path.join(__dirname, '../test-fixtures/test-paper.pdf')

    // Find the error demo section upload input (4th one)
    const uploadInputs = page.getByLabelText(/upload.*paper/i)
    const errorInput = uploadInputs.nth(3)

    // Upload file (should fail in demo)
    await errorInput.setInputFiles(testFile)

    // Check error appears
    await expect(page.getByText(/demo error/i)).toBeVisible({ timeout: 5000 })

    // Check Try Again button appears
    await expect(page.getByRole('button', { name: /try again/i })).toBeVisible()
  })
})
