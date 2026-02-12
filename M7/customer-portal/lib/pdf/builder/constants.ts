export const DELIVEROO_LOGO_PATH = '/deliveroo-pdf-logo.png'

export const FOOTER_LINES = [
  'Deliveroo Logistics | ul. Logistyczna 123, 00-001 Warsaw, Poland',
  'Phone: +48 123 456 789 | Email: contact@deliveroo.pl'
] as const

export const LAYOUT = {
  margin: 20,
  labelColumnX: 20,
  valueColumnX: 80,
  contentWidth: 170,
  sectionHeaderHeight: 8,
  rowHeight: 8,
  sectionSpacing: 15,
  footerTopMargin: 25,
  minBottomMargin: 30
} as const

export function formatDate(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date
  return new Intl.DateTimeFormat('en-US', {
    month: 'long',
    day: 'numeric',
    year: 'numeric'
  }).format(d)
}

export function formatCurrency(amount: number, currency: string = 'EUR'): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: currency || 'EUR'
  }).format(amount)
}

export function formatLabel(value: string): string {
  return value
    ? value.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, (l) => l.toUpperCase())
    : 'Not specified'
}

export async function loadLogo(): Promise<string | null> {
  try {
    const response = await fetch(DELIVEROO_LOGO_PATH)
    const blob = await response.blob()
    return await new Promise((resolve) => {
      const reader = new FileReader()
      reader.onloadend = () => resolve(reader.result as string)
      reader.readAsDataURL(blob)
    })
  } catch (err) {
    console.error('Failed to load logo for PDF', err)
    return null
  }
}
