import { test, expect } from '@playwright/test'

test.describe('Paper Details Page E2E', () => {
  const mockPaperData = {
    paper_id: 'test-paper-123',
    metadata: {
      title: 'Advanced Machine Learning Techniques',
      authors: ['John Doe', 'Jane Smith'],
      abstract: 'This paper presents novel approaches to machine learning optimization.',
    },
    scores: {
      overall: 85,
      novelty: {
        category: 'novelty',
        score: 90,
        confidence: 0.85,
        explanation: 'The paper introduces innovative methodologies.',
      },
      technical_quality: {
        category: 'technical_quality',
        score: 80,
        confidence: 0.9,
        explanation: 'Strong technical foundation with rigorous analysis.',
      },
      clarity: {
        category: 'clarity',
        score: 85,
        confidence: 0.8,
        explanation: 'Well-structured and clearly written.',
      },
      impact: {
        category: 'impact',
        score: 85,
        confidence: 0.75,
        explanation: 'Significant potential for real-world applications.',
      },
    },
    recommendations: [
      {
        id: 'rec-1',
        category: 'technical_quality',
        priority: 'high',
        title: 'Expand experimental validation',
        description: 'Add more comprehensive experiments across diverse datasets.',
        impact: 'Would significantly strengthen the empirical evidence.',
      },
      {
        id: 'rec-2',
        category: 'clarity',
        priority: 'medium',
        title: 'Improve figure captions',
        description: 'Add more detailed explanations to figures.',
        impact: 'Would enhance reader comprehension.',
      },
    ],
  }

  test.beforeEach(async ({ page }) => {
    // Mock the API endpoint
    await page.route('**/api/papers/test-paper-123', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockPaperData),
      })
    })

    // Mock WebSocket connection
    await page.route('**/ws/papers/test-paper-123', async (route) => {
      await route.fulfill({
        status: 101,
        headers: {
          'Upgrade': 'websocket',
          'Connection': 'Upgrade',
        },
      })
    })

    await page.goto('/papers/test-paper-123')
  })

  test('should display paper metadata', async ({ page }) => {
    // Check paper title
    await expect(page.getByRole('heading', { name: /advanced machine learning techniques/i })).toBeVisible()

    // Check authors
    await expect(page.getByText('John Doe, Jane Smith')).toBeVisible()

    // Check abstract
    await expect(page.getByText(/novel approaches to machine learning/i)).toBeVisible()
  })

  test('should display tab navigation', async ({ page }) => {
    // Check all tabs are visible
    await expect(page.getByRole('tab', { name: /overview/i })).toBeVisible()
    await expect(page.getByRole('tab', { name: /scores/i })).toBeVisible()
    await expect(page.getByRole('tab', { name: /recommendations/i })).toBeVisible()
  })

  test('should display analysis scores', async ({ page }) => {
    // Check overall score
    await expect(page.getByText(/overall 85/i)).toBeVisible()

    // Check category scores
    await expect(page.getByText(/novelty/i)).toBeVisible()
    await expect(page.getByText(/technical quality/i)).toBeVisible()
    await expect(page.getByText(/clarity/i)).toBeVisible()
    await expect(page.getByText(/impact/i)).toBeVisible()

    // Check scores are displayed
    await expect(page.getByText('90')).toBeVisible() // novelty score
    await expect(page.getByText('80')).toBeVisible() // technical quality score
  })

  test('should display confidence indicators', async ({ page }) => {
    // Check confidence percentages
    await expect(page.getByText(/85% confidence/i)).toBeVisible() // novelty
    await expect(page.getByText(/90% confidence/i)).toBeVisible() // technical quality
  })

  test('should expand category explanations on click', async ({ page }) => {
    // Initially, explanations should be hidden
    await expect(page.getByText(/innovative methodologies/i)).not.toBeVisible()

    // Click Show More
    const showMoreButtons = page.getByRole('button', { name: /show more/i })
    await showMoreButtons.first().click()

    // Explanation should now be visible
    await expect(page.getByText(/innovative methodologies/i)).toBeVisible()

    // Click Show Less
    await page.getByRole('button', { name: /show less/i }).first().click()

    // Explanation should be hidden again
    await expect(page.getByText(/innovative methodologies/i)).not.toBeVisible()
  })

  test('should display recommendations panel', async ({ page }) => {
    // Check panel title
    await expect(page.getByRole('heading', { name: /recommendations/i })).toBeVisible()

    // Check recommendation count
    await expect(page.getByText(/2 recommendations/i)).toBeVisible()

    // Check recommendations are displayed
    await expect(page.getByText(/expand experimental validation/i)).toBeVisible()
    await expect(page.getByText(/improve figure captions/i)).toBeVisible()
  })

  test('should display recommendation priorities', async ({ page }) => {
    // Check priority badges
    await expect(page.getByText(/high/i).first()).toBeVisible()
    await expect(page.getByText(/medium/i).first()).toBeVisible()
  })

  test('should filter recommendations by category', async ({ page }) => {
    // Initially all recommendations visible
    await expect(page.getByText(/expand experimental validation/i)).toBeVisible()
    await expect(page.getByText(/improve figure captions/i)).toBeVisible()

    // Click Technical filter
    await page.getByRole('button', { name: /filter by technical/i }).click()

    // Only technical recommendation should be visible
    await expect(page.getByText(/expand experimental validation/i)).toBeVisible()
    await expect(page.getByText(/improve figure captions/i)).not.toBeVisible()

    // Click Clarity filter
    await page.getByRole('button', { name: /filter by clarity/i }).click()

    // Only clarity recommendation should be visible
    await expect(page.getByText(/expand experimental validation/i)).not.toBeVisible()
    await expect(page.getByText(/improve figure captions/i)).toBeVisible()

    // Click All filter
    await page.getByRole('button', { name: /filter by all/i }).click()

    // Both recommendations should be visible again
    await expect(page.getByText(/expand experimental validation/i)).toBeVisible()
    await expect(page.getByText(/improve figure captions/i)).toBeVisible()
  })

  test('should expand recommendation details on click', async ({ page }) => {
    // Initially, details should be hidden
    await expect(page.getByText(/add more comprehensive experiments/i)).not.toBeVisible()

    // Click Show More Details
    const showMoreButtons = page.getByRole('button', { name: /show more details/i })
    await showMoreButtons.first().click()

    // Details should now be visible
    await expect(page.getByText(/add more comprehensive experiments/i)).toBeVisible()

    // Click Collapse
    await page.getByRole('button', { name: /collapse/i }).first().click()

    // Details should be hidden again
    await expect(page.getByText(/add more comprehensive experiments/i)).not.toBeVisible()
  })

  test('should export recommendations', async ({ page }) => {
    // Set up download listener
    const downloadPromise = page.waitForEvent('download')

    // Click Export button
    await page.getByRole('button', { name: /export/i }).click()

    // Wait for download
    const download = await downloadPromise

    // Check filename
    expect(download.suggestedFilename()).toBe('recommendations.txt')
  })

  test('should switch between tabs', async ({ page }) => {
    // Initially on Overview tab (both panels visible)
    await expect(page.getByRole('heading', { name: /analysis scores/i })).toBeVisible()
    await expect(page.getByRole('heading', { name: /recommendations/i })).toBeVisible()

    // Click Scores tab
    await page.getByRole('tab', { name: /^scores$/i }).click()

    // Only scores panel visible
    await expect(page.getByRole('heading', { name: /analysis scores/i })).toBeVisible()
    const recommendationsHeadings = page.getByRole('heading', { name: /recommendations/i })
    await expect(recommendationsHeadings).toHaveCount(0)

    // Click Recommendations tab
    await page.getByRole('tab', { name: /^recommendations$/i }).click()

    // Only recommendations panel visible
    await expect(page.getByRole('heading', { name: /recommendations/i })).toBeVisible()
    const scoresHeadings = page.getByRole('heading', { name: /analysis scores/i })
    await expect(scoresHeadings).toHaveCount(0)

    // Click Overview tab
    await page.getByRole('tab', { name: /overview/i }).click()

    // Both panels visible again
    await expect(page.getByRole('heading', { name: /analysis scores/i })).toBeVisible()
    await expect(page.getByRole('heading', { name: /recommendations/i })).toBeVisible()
  })

  test('should handle paper not found error', async ({ page }) => {
    // Override mock to return 404
    await page.route('**/api/papers/nonexistent-paper', async (route) => {
      await route.fulfill({
        status: 404,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Paper not found' }),
      })
    })

    await page.goto('/papers/nonexistent-paper')

    // Check error message
    await expect(page.getByRole('heading', { name: /paper not found/i })).toBeVisible()
  })

  test('should handle API error gracefully', async ({ page }) => {
    // Override mock to return 500
    await page.route('**/api/papers/error-paper', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Internal server error' }),
      })
    })

    await page.goto('/papers/error-paper')

    // Check error message
    await expect(page.getByRole('heading', { name: /error loading paper/i })).toBeVisible()
  })

  test('should display loading state', async ({ page }) => {
    // Override mock to delay response
    await page.route('**/api/papers/slow-paper', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 1000))
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockPaperData),
      })
    })

    await page.goto('/papers/slow-paper')

    // Check loading skeletons
    await expect(page.getByText(/loading/i)).toBeVisible()

    // Wait for content to load
    await expect(page.getByRole('heading', { name: /advanced machine learning techniques/i })).toBeVisible({ timeout: 3000 })
  })

  test('should display empty state for no scores', async ({ page }) => {
    // Mock paper with no scores
    const noScoresPaper = {
      ...mockPaperData,
      scores: {
        overall: 0,
        novelty: { category: 'novelty', score: 0, confidence: 0 },
        technical_quality: { category: 'technical_quality', score: 0, confidence: 0 },
        clarity: { category: 'clarity', score: 0, confidence: 0 },
        impact: { category: 'impact', score: 0, confidence: 0 },
      },
    }

    await page.route('**/api/papers/no-scores', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(noScoresPaper),
      })
    })

    await page.goto('/papers/no-scores')

    // Check empty state message
    await expect(page.getByText(/pending/i)).toBeVisible()
  })

  test('should display empty state for no recommendations', async ({ page }) => {
    // Mock paper with no recommendations
    const noRecsPaper = {
      ...mockPaperData,
      recommendations: [],
    }

    await page.route('**/api/papers/no-recs', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(noRecsPaper),
      })
    })

    await page.goto('/papers/no-recs')

    // Check empty state message
    await expect(page.getByText(/no recommendations available/i)).toBeVisible()
  })

  test('should have accessible ARIA attributes', async ({ page }) => {
    // Check region labels
    await expect(page.getByRole('region', { name: /overall score/i })).toBeVisible()

    // Check list labels
    await expect(page.getByRole('list', { name: /category scores/i })).toBeVisible()
    await expect(page.getByRole('list', { name: /recommendations/i })).toBeVisible()

    // Check filter group
    await expect(page.getByRole('group', { name: /filter recommendations/i })).toBeVisible()
  })
})
