import { test, expect, type Page } from '@playwright/test'

/** Wait for Vue/Nuxt hydration to complete before interacting with reactive elements */
async function waitForHydration(page: Page) {
  await page.waitForLoadState('networkidle')
}

test.describe('New Transportation Request', () => {
  test.use({ storageState: 'e2e/.auth/user.json' })

  test('should display the new transportation request form', async ({ page }) => {
    await page.goto('/dashboard/transportation/new')
    await expect(page.getByRole('heading', { name: 'New Transportation Request' })).toBeVisible()
    await expect(page.getByText('Create a new road transportation request across Europe')).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Service Type' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Next' })).toBeVisible()
  })

  test('should require service type before proceeding', async ({ page }) => {
    await page.goto('/dashboard/transportation/new')
    await expect(page.getByRole('heading', { name: 'New Transportation Request' })).toBeVisible()

    const nextButton = page.getByRole('button', { name: 'Next' })
    await expect(nextButton).toBeDisabled()
  })

  test('should complete full transportation request flow', async ({ page }) => {
    await page.goto('/dashboard/transportation/new')
    await waitForHydration(page)
    await expect(page.getByRole('heading', { name: 'New Transportation Request' })).toBeVisible()

    // Step 1: Service Type
    await page.getByText('Full Truckload (FTL)').click()
    await page.getByRole('button', { name: 'Next' }).click()

    // Step 2: Pickup Information
    await expect(page.getByRole('heading', { name: 'Pickup Information' })).toBeVisible()
    await page.getByPlaceholder('Enter pickup address').fill('ul. Logistyczna 10')
    await page.getByPlaceholder('Enter city').fill('Warsaw')
    await page.locator('select').first().selectOption('Poland')
    await page.getByPlaceholder('Contact person name').fill('Jan Kowalski')
    await page.getByPlaceholder('Phone number').fill('+48123456789')
    await page.locator('input[type="date"]').first().fill('2025-03-15')
    await page.getByRole('button', { name: 'Next' }).click()

    // Step 3: Delivery Information
    await expect(page.getByRole('heading', { name: 'Delivery Information' })).toBeVisible()
    await page.getByPlaceholder('Enter delivery address').fill('Hauptstrasse 20')
    await page.getByPlaceholder('Enter city').fill('Berlin')
    await page.locator('select').first().selectOption('Germany')
    await page.getByPlaceholder('Contact person name').fill('Hans Mueller')
    await page.getByPlaceholder('Phone number').fill('+49301234567')
    await page.getByRole('button', { name: 'Next' }).click()

    // Step 4: Cargo Information
    await expect(page.getByRole('heading', { name: 'Cargo Information' })).toBeVisible()
    await page.getByPlaceholder('Describe the cargo to be transported').fill('Industrial machinery parts')
    await page.getByPlaceholder('Enter weight in kg').fill('1500')
    await page.getByRole('button', { name: 'Next' }).click()

    // Step 5: Special Instructions (optional - can proceed with empty)
    await expect(page.getByRole('heading', { name: 'Special Instructions' })).toBeVisible()
    await page.getByRole('button', { name: 'Next' }).click()

    // Step 6: Review & Submit
    await expect(page.getByRole('heading', { name: 'Review & Submit' })).toBeVisible()
    await page.getByRole('button', { name: 'Submit Request' }).click()

    // Success modal
    await expect(page.getByText('Transportation Request Submitted Successfully!')).toBeVisible({ timeout: 10000 })
    await expect(page.getByText('Reference Number')).toBeVisible()
    await expect(page.getByText(/TR-\d{4}-\d+/)).toBeVisible()
  })

  test('should allow navigation back between steps', async ({ page }) => {
    await page.goto('/dashboard/transportation/new')
    await waitForHydration(page)
    await expect(page.getByRole('heading', { name: 'New Transportation Request' })).toBeVisible()

    // Step 1 -> 2
    await page.getByText('Less Than Truckload (LTL)').click()
    await page.getByRole('button', { name: 'Next' }).click()

    await expect(page.getByRole('heading', { name: 'Pickup Information' })).toBeVisible()

    // Step 2 -> 1 (Back)
    await page.getByRole('button', { name: 'Back' }).click()
    await expect(page.getByRole('heading', { name: 'Service Type' })).toBeVisible()
    await expect(page.locator('input[value="LESS_THAN_TRUCKLOAD"]')).toBeChecked()
  })

})

test.describe('New Transportation Request - Auth', () => {
  test.use({ storageState: { cookies: [], origins: [] } })

  test('should redirect unauthenticated user to login', async ({ page }) => {
    await page.goto('/login')
    const authKeys = ['auth_user', 'auth_company', 'auth_isAuthenticated']
    await page.evaluate((keys) => {
      keys.forEach((k) => localStorage.removeItem(k))
    }, authKeys)

    await page.goto('/dashboard/transportation/new')

    await expect(page).toHaveURL(/\/login/)
  })
})
