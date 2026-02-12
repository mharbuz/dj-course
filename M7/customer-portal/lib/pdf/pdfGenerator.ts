import { PdfDocumentBuilder } from './builder/PdfDocumentBuilder'
import { KeyValueSection } from './sections/KeyValueSection'
import { formatCurrency, formatDate } from './builder/constants'

interface Address {
  street: string
  city: string
  postalCode?: string
  country: string
}

interface TransportationRequestData {
  id: string
  requestNumber: string
  status: string
  priority: string
  pickupLocation: {
    address: Address
    contactPerson: string
    contactPhone: string
    contactEmail: string
  }
  deliveryLocation: {
    address: Address
    contactPerson: string
    contactPhone: string
    contactEmail: string
  }
  cargo: {
    description: string
    cargoType: string
    weight: number
    dimensions: { length: number; width: number; height: number; unit: string }
    value: number
    currency: string
    packaging: string
    quantity: number
    unitType: string
  }
  serviceType: string
  vehicleRequirements?: { vehicleType: string; capacity: number }
  requestedPickupDate: Date | string
  requestedDeliveryDate: Date | string
  specialInstructions?: string
  requiresInsurance: boolean
  requiresCustomsClearance: boolean
  estimatedCost?: number
  finalCost?: number
  currency: string
  trackingNumber?: string
  createdAt: Date | string
}

interface WarehousingRequestData {
  id: string
  requestNumber: string
  status: string
  priority: string
  storageType: string
  estimatedVolume: number
  estimatedWeight: number
  cargo: {
    description: string
    cargoType: string
    weight: number
    dimensions: { length: number; width: number; height: number; unit: string }
    value: number
    currency: string
    packaging: string
    quantity: number
    unitType: string
  }
  estimatedStorageDuration: { value: number; unit: string }
  plannedStartDate: Date | string
  plannedEndDate?: Date | string
  handlingServices: string[]
  valueAddedServices: string[]
  securityLevel: string
  requiresTemperatureControl: boolean
  requiresHumidityControl: boolean
  requiresSpecialHandling: boolean
  specialInstructions?: string
  estimatedCost?: number
  finalCost?: number
  currency: string
  billingType: string
  storageLocation?: string
  createdAt: Date | string
}

function formatAddress(addr: Address): string {
  const parts = [addr.street, addr.city, addr.postalCode, addr.country].filter(Boolean)
  return parts.join(', ')
}

export const PDFGenerator = {
  async generateTransportationRequestPDF(request: TransportationRequestData): Promise<void> {
    const cargoDim = request.cargo.dimensions
    const cargo = request.cargo

    const builder = await PdfDocumentBuilder.create({
      title: 'Transportation Request',
      filename: `Transportation_Request_${request.requestNumber}.pdf`
    })

    builder
      .addHeader()
      .addSection(
        new KeyValueSection('Request Information', [
          { label: 'Request Number', value: request.requestNumber },
          { label: 'Status', value: request.status },
          { label: 'Priority', value: request.priority },
          { label: 'Created', value: formatDate(request.createdAt) }
        ])
      )
      .addSection(
        new KeyValueSection('Pickup Location', [
          { label: 'Address', value: formatAddress(request.pickupLocation.address), fullWidth: true },
          { label: 'Contact Person', value: request.pickupLocation.contactPerson },
          { label: 'Phone', value: request.pickupLocation.contactPhone },
          { label: 'Email', value: request.pickupLocation.contactEmail }
        ])
      )
      .addSection(
        new KeyValueSection('Delivery Location', [
          { label: 'Address', value: formatAddress(request.deliveryLocation.address), fullWidth: true },
          { label: 'Contact Person', value: request.deliveryLocation.contactPerson },
          { label: 'Phone', value: request.deliveryLocation.contactPhone },
          { label: 'Email', value: request.deliveryLocation.contactEmail }
        ])
      )
      .addSection(
        new KeyValueSection('Cargo Information', [
          { label: 'Description', value: cargo.description, fullWidth: true },
          { label: 'Cargo Type', value: cargo.cargoType },
          { label: 'Weight', value: `${cargo.weight} kg` },
          {
            label: 'Dimensions',
            value: `${cargoDim.length} × ${cargoDim.width} × ${cargoDim.height} ${cargoDim.unit}`
          },
          { label: 'Quantity', value: `${cargo.quantity} ${cargo.unitType}` },
          { label: 'Value', value: formatCurrency(cargo.value, cargo.currency) },
          { label: 'Packaging', value: cargo.packaging }
        ])
      )
      .addSection(
        new KeyValueSection('Service Details', [
          { label: 'Service Type', value: request.serviceType },
          { label: 'Requested Pickup Date', value: formatDate(request.requestedPickupDate) },
          { label: 'Requested Delivery Date', value: formatDate(request.requestedDeliveryDate) },
          ...(request.vehicleRequirements
            ? [{ label: 'Vehicle Type', value: request.vehicleRequirements.vehicleType }]
            : []),
          { label: 'Requires Insurance', value: request.requiresInsurance ? 'Yes' : 'No' },
          { label: 'Requires Customs Clearance', value: request.requiresCustomsClearance ? 'Yes' : 'No' },
          ...(request.specialInstructions
            ? [{ label: 'Special Instructions', value: request.specialInstructions, fullWidth: true }]
            : []),
          ...(request.trackingNumber
            ? [{ label: 'Tracking Number', value: request.trackingNumber }]
            : [])
        ].filter(Boolean) as { label: string; value: string; fullWidth?: boolean }[])
      )
      .addSection(
        new KeyValueSection('Pricing', [
          ...(request.estimatedCost
            ? [{ label: 'Estimated Cost', value: formatCurrency(request.estimatedCost, request.currency) }]
            : []),
          ...(request.finalCost
            ? [{ label: 'Final Cost', value: formatCurrency(request.finalCost, request.currency) }]
            : [])
        ].filter(Boolean) as { label: string; value: string }[])
      )

    await builder.build()
  },

  async generateWarehousingRequestPDF(request: WarehousingRequestData): Promise<void> {
    const cargo = request.cargo
    const dim = cargo.dimensions

    const builder = await PdfDocumentBuilder.create({
      title: 'Warehousing Request',
      filename: `Warehousing_Request_${request.requestNumber}.pdf`
    })

    builder
      .addHeader()
      .addSection(
        new KeyValueSection('Request Information', [
          { label: 'Request Number', value: request.requestNumber },
          { label: 'Status', value: request.status },
          { label: 'Priority', value: request.priority },
          { label: 'Created', value: formatDate(request.createdAt) }
        ])
      )
      .addSection(
        new KeyValueSection('Storage Information', [
          { label: 'Storage Type', value: request.storageType },
          { label: 'Estimated Volume', value: `${request.estimatedVolume} m³` },
          { label: 'Estimated Weight', value: `${request.estimatedWeight} kg` },
          { label: 'Security Level', value: request.securityLevel },
          ...(request.storageLocation
            ? [{ label: 'Storage Location', value: request.storageLocation }]
            : []),
          { label: 'Planned Start Date', value: formatDate(request.plannedStartDate) },
          ...(request.plannedEndDate
            ? [{ label: 'Planned End Date', value: formatDate(request.plannedEndDate) }]
            : []),
          {
            label: 'Storage Duration',
            value: `${request.estimatedStorageDuration.value} ${request.estimatedStorageDuration.unit}`
          }
        ].filter(Boolean) as { label: string; value: string }[])
      )
      .addSection(
        new KeyValueSection('Cargo Information', [
          { label: 'Description', value: cargo.description, fullWidth: true },
          { label: 'Cargo Type', value: cargo.cargoType },
          { label: 'Weight', value: `${cargo.weight} kg` },
          {
            label: 'Dimensions',
            value: `${dim.length} × ${dim.width} × ${dim.height} ${dim.unit}`
          },
          { label: 'Quantity', value: `${cargo.quantity} ${cargo.unitType}` },
          { label: 'Value', value: formatCurrency(cargo.value, cargo.currency) },
          { label: 'Packaging', value: cargo.packaging }
        ])
      )
      .addSection(
        new KeyValueSection('Service Requirements', [
          ...(request.handlingServices.length > 0
            ? [{ label: 'Handling Services', value: request.handlingServices.join(', ') }]
            : []),
          ...(request.valueAddedServices.length > 0
            ? [{ label: 'Value Added Services', value: request.valueAddedServices.join(', ') }]
            : []),
          { label: 'Requires Temperature Control', value: request.requiresTemperatureControl ? 'Yes' : 'No' },
          { label: 'Requires Humidity Control', value: request.requiresHumidityControl ? 'Yes' : 'No' },
          { label: 'Requires Special Handling', value: request.requiresSpecialHandling ? 'Yes' : 'No' },
          ...(request.specialInstructions
            ? [{ label: 'Special Instructions', value: request.specialInstructions, fullWidth: true }]
            : [])
        ].filter(Boolean) as { label: string; value: string; fullWidth?: boolean }[])
      )
      .addSection(
        new KeyValueSection('Pricing', [
          { label: 'Billing Type', value: request.billingType },
          ...(request.estimatedCost
            ? [{ label: 'Estimated Cost', value: formatCurrency(request.estimatedCost, request.currency) }]
            : []),
          ...(request.finalCost
            ? [{ label: 'Final Cost', value: formatCurrency(request.finalCost, request.currency) }]
            : [])
        ].filter(Boolean) as { label: string; value: string }[])
      )

    await builder.build()
  }
}
