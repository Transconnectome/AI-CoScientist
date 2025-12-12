import { test, expect } from '@playwright/test'

test.describe('Basic Navigation', () => {
  test('should load homepage', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('h1')).toContainText('AI-CoScientist Dashboard')
  })
})
