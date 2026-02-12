import { PdfDocumentBuilder } from './builder/PdfDocumentBuilder'
import { KeyValueSection } from './sections/KeyValueSection'
import { formatCurrency, formatDate, formatLabel } from './builder/constants'

interface TransportationRequestFormData {
  serviceType: string
  pickupLocation: {
    address: { street: string; city: string; country: string }
    contactPerson: string
    contactPhone: string
    contactEmail?: string
    loadingType?: string
  }
  deliveryLocation: {
    address: { street: string; city: string; country: string }
    contactPerson: string
    contactPhone: string
    contactEmail?: string
    loadingType?: string
  }
  cargo: {
    description: string
    cargoType: string
    weight: number
    packaging: string
    quantity: number
    unitType: string
    value: number
    currency: string
    fragile?: boolean
    stackable?: boolean
  }
  requestedPickupDate: string | Date
  requestedDeliveryDate?: string | Date
  specialInstructions?: string
  requiresInsurance: boolean
  requiresCustomsClearance: boolean
  priority: string
  currency: string
}

interface TransportationRequestPdfOptions {
  requestNumber?: string
  createdAt?: Date | string
}

function formatAddress(addr: { street: string; city: string; country: string }): string {
  return `${addr.street}, ${addr.city}, ${addr.country}`
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

export async function generateTransportationRequestPDF(
  formData: TransportationRequestFormData,
  options: TransportationRequestPdfOptions = {}
): Promise<void> {
  const filename = options.requestNumber
    ? `Transportation_Request_${options.requestNumber}.pdf`
    : `Transportation_Request_${new Date().toISOString().split('T')[0]}.pdf`

  const requestInfoRows: { label: string; value: string }[] = [
    ...(options.requestNumber ? [{ label: 'Request Number', value: options.requestNumber }] : []),
    { label: 'Service Type', value: formatLabel(formData.serviceType) },
    { label: 'Priority', value: formatLabel(formData.priority) },
    ...(options.createdAt ? [{ label: 'Created', value: safeFormatDate(options.createdAt) }] : [])
  ]

  const pickupRows: { label: string; value: string; fullWidth?: boolean }[] = [
    { label: 'Address', value: formatAddress(formData.pickupLocation.address), fullWidth: true },
    { label: 'Contact Person', value: formData.pickupLocation.contactPerson },
    { label: 'Phone', value: formData.pickupLocation.contactPhone },
    ...(formData.pickupLocation.contactEmail ? [{ label: 'Email', value: formData.pickupLocation.contactEmail }] : []),
    { label: 'Requested Pickup Date', value: safeFormatDate(formData.requestedPickupDate) },
    ...(formData.pickupLocation.loadingType
      ? [{ label: 'Loading Type', value: formatLabel(formData.pickupLocation.loadingType) }]
      : [])
  ]

  const deliveryRows: { label: string; value: string; fullWidth?: boolean }[] = [
    { label: 'Address', value: formatAddress(formData.deliveryLocation.address), fullWidth: true },
    { label: 'Contact Person', value: formData.deliveryLocation.contactPerson },
    { label: 'Phone', value: formData.deliveryLocation.contactPhone },
    ...(formData.deliveryLocation.contactEmail ? [{ label: 'Email', value: formData.deliveryLocation.contactEmail }] : []),
    ...(formData.requestedDeliveryDate
      ? [{ label: 'Requested Delivery Date', value: safeFormatDate(formData.requestedDeliveryDate) }]
      : []),
    ...(formData.deliveryLocation.loadingType
      ? [{ label: 'Unloading Type', value: formatLabel(formData.deliveryLocation.loadingType) }]
      : [])
  ]

  const cargoRows: { label: string; value: string; fullWidth?: boolean }[] = [
    { label: 'Description', value: formData.cargo.description, fullWidth: true },
    { label: 'Cargo Type', value: formatLabel(formData.cargo.cargoType) },
    { label: 'Weight', value: `${formData.cargo.weight} kg` },
    { label: 'Packaging', value: formatLabel(formData.cargo.packaging) },
    { label: 'Quantity', value: `${formData.cargo.quantity} ${formData.cargo.unitType}` },
    ...(formData.cargo.value > 0
      ? [{ label: 'Estimated Value', value: formatCurrency(formData.cargo.value, formData.cargo.currency) }]
      : []),
    ...(formData.cargo.fragile !== undefined
      ? [{ label: 'Fragile', value: formData.cargo.fragile ? 'Yes' : 'No' }]
      : []),
    ...(formData.cargo.stackable !== undefined
      ? [{ label: 'Stackable', value: formData.cargo.stackable ? 'Yes' : 'No' }]
      : [])
  ]

  const serviceRows: { label: string; value: string; fullWidth?: boolean }[] = [
    { label: 'Requires Insurance', value: formData.requiresInsurance ? 'Yes' : 'No' },
    { label: 'Requires Customs Clearance', value: formData.requiresCustomsClearance ? 'Yes' : 'No' },
    ...(formData.specialInstructions
      ? [{ label: 'Special Instructions', value: formData.specialInstructions, fullWidth: true }]
      : [])
  ]

  const builder = await PdfDocumentBuilder.create({
    title: 'Transportation Request',
    filename
  })

  builder
    .addHeader()
    .addSection(new KeyValueSection('Request Information', requestInfoRows))
    .addSection(new KeyValueSection('Pickup Location', pickupRows))
    .addSection(new KeyValueSection('Delivery Location', deliveryRows))
    .addSection(new KeyValueSection('Cargo Information', cargoRows))
    .addSection(new KeyValueSection('Service Requirements', serviceRows))

  await builder.build()
}
