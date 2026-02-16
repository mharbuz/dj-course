import { test as setup } from '@playwright/test'

const authFile = 'e2e/.auth/user.json'

setup('authenticate', async ({ page }) => {
  await page.goto('/login')

  // Set auth state directly in localStorage (bypasses flaky login flow)
  await page.evaluate(() => {
    const user = {
      id: '1',
      email: 'test@example.com',
      firstName: 'John',
      lastName: 'Doe',
      phone: '+48123456789',
      role: 'COMPANY_ADMIN',
      companyId: '1',
      permissions: ['CREATE_REQUEST', 'VIEW_REQUEST', 'EDIT_REQUEST', 'MANAGE_TEAM'],
      isActive: true,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    }
    const company = {
      id: '1',
      name: 'Example Logistics Ltd.',
      registrationNumber: 'PL1234567890',
      vatNumber: 'PL1234567890',
      address: { street: 'ul. Logistyczna 123', city: 'Warsaw', postalCode: '00-001', country: 'Poland' },
      contactInfo: {
        primaryEmail: 'contact@example.com',
        primaryPhone: '+48123456789',
        emergencyContact: { name: 'Emergency Contact', phone: '+48987654321', email: 'emergency@example.com', relationship: 'Manager' },
      },
      billingAddress: { street: 'ul. Logistyczna 123', city: 'Warsaw', postalCode: '00-001', country: 'Poland' },
      creditLimit: 50000,
      creditUsed: 15000,
      industryType: 'Manufacturing',
      employees: [],
      paymentTerms: '30 days',
      isActive: true,
      createdAt: new Date().toISOString(),
    }
    localStorage.setItem('auth_user', JSON.stringify(user))
    localStorage.setItem('auth_company', JSON.stringify(company))
    localStorage.setItem('auth_isAuthenticated', 'true')
  })

  // Reload so auth plugin picks up localStorage
  await page.reload()
  await page.waitForURL(/\/login/)

  // Navigate to dashboard (middleware will pass with auth from localStorage)
  await page.goto('/dashboard')
  await page.waitForURL(/\/dashboard/, { waitUntil: 'commit', timeout: 10000 })

  await page.context().storageState({ path: authFile })
})
