import { PdfDocumentBuilder } from './builder/PdfDocumentBuilder'
import { KeyValueSection } from './sections/KeyValueSection'
import { formatCurrency, formatDate, formatLabel } from './builder/constants'

interface WarehousingRequestFormData {
  storageType: string
  securityLevel: string
  estimatedVolume: number
  estimatedWeight: number
  estimatedStorageDuration: {
    value: number
    unit: 'days' | 'weeks' | 'months' | 'years'
  }
  plannedStartDate: string | Date
  plannedEndDate?: string | Date
  handlingServices: string[]
  valueAddedServices: string[]
  requiresTemperatureControl: boolean
  requiresHumidityControl: boolean
  requiresSpecialHandling: boolean
  specialInstructions?: string
  billingType: string
  cargo: {
    description: string
    cargoType: string
    packaging: string
    quantity: number
    unitType: string
    value: number
    currency: string
  }
  priority: string
}

interface WarehousingRequestPdfOptions {
  requestNumber?: string
  createdAt?: Date | string
  storageLocation?: string
}

function safeFormatDate(value: string | Date | undefined): string {
  if (!value) return 'Not specified'
  try {
    const d = typeof value === 'string' ? new Date(value) : value
    return formatDate(d)
  } catch {
    return 'Not specified'
  }
}

export async function generateWarehousingRequestPDF(
  formData: WarehousingRequestFormData,
  options: WarehousingRequestPdfOptions = {}
): Promise<void> {
  const duration = formData.estimatedStorageDuration || { value: 0, unit: 'months' }
  const filename = options.requestNumber
    ? `Warehousing_Request_${options.requestNumber}.pdf`
    : `Warehousing_Request_${new Date().toISOString().split('T')[0]}.pdf`

  const requestInfoRows: { label: string; value: string }[] = [
    ...(options.requestNumber ? [{ label: 'Request Number', value: options.requestNumber }] : []),
    { label: 'Storage Type', value: formatLabel(formData.storageType) },
    { label: 'Priority', value: formatLabel(formData.priority) },
    ...(options.createdAt ? [{ label: 'Created', value: safeFormatDate(options.createdAt) }] : [])
  ]

  const storageInfoRows: { label: string; value: string }[] = [
    { label: 'Estimated Volume', value: `${formData.estimatedVolume} m³` },
    { label: 'Estimated Weight', value: `${formData.estimatedWeight} kg` },
    { label: 'Security Level', value: formatLabel(formData.securityLevel) },
    ...(options.storageLocation ? [{ label: 'Storage Location', value: options.storageLocation }] : []),
    { label: 'Planned Start Date', value: safeFormatDate(formData.plannedStartDate) },
    ...(formData.plannedEndDate ? [{ label: 'Planned End Date', value: safeFormatDate(formData.plannedEndDate) }] : []),
    { label: 'Storage Duration', value: `${duration.value} ${duration.unit}` },
    { label: 'Billing Type', value: formatLabel(formData.billingType) }
  ]

  const cargoInfoRows: { label: string; value: string; fullWidth?: boolean }[] = [
    {
      label: 'Description',
      value: formData.cargo?.description || 'No description provided',
      fullWidth: true
    },
    { label: 'Cargo Type', value: formatLabel(formData.cargo?.cargoType) },
    { label: 'Packaging', value: formatLabel(formData.cargo?.packaging) },
    { label: 'Quantity', value: `${formData.cargo?.quantity ?? 0} ${formData.cargo?.unitType ?? ''}` },
    ...(formData.cargo?.value && formData.cargo.value > 0
      ? [{ label: 'Estimated Value', value: formatCurrency(formData.cargo.value, formData.cargo.currency) }]
      : [])
  ]

  const serviceRows: { label: string; value: string; fullWidth?: boolean }[] = [
    ...(formData.handlingServices?.length
      ? [{ label: 'Handling Services', value: formData.handlingServices.map(formatLabel).join(', ') }]
      : []),
    ...(formData.valueAddedServices?.length
      ? [{ label: 'Value Added Services', value: formData.valueAddedServices.map(formatLabel).join(', ') }]
      : []),
    { label: 'Requires Temperature Control', value: formData.requiresTemperatureControl ? 'Yes' : 'No' },
    { label: 'Requires Humidity Control', value: formData.requiresHumidityControl ? 'Yes' : 'No' },
    { label: 'Requires Special Handling', value: formData.requiresSpecialHandling ? 'Yes' : 'No' },
    ...(formData.specialInstructions
      ? [{ label: 'Special Instructions', value: formData.specialInstructions, fullWidth: true }]
      : [])
  ]

  const builder = await PdfDocumentBuilder.create({
    title: 'Warehousing Request',
    filename
  })

  builder
    .addHeader()
    .addSection(new KeyValueSection('Request Information', requestInfoRows))
    .addSection(new KeyValueSection('Storage Information', storageInfoRows))
    .addSection(new KeyValueSection('Cargo Information', cargoInfoRows))
    .addSection(new KeyValueSection('Service Requirements', serviceRows))

  await builder.build()
}
