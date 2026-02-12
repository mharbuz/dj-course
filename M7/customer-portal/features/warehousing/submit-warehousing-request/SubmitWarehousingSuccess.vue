<template>
  <SuccessModal
    :show="store.showSuccessModal"
    title="Warehousing Request Submitted Successfully!"
    message="Your warehousing request has been received and our storage specialists are reviewing your requirements."
    :reference-number="store.generatedRequestNumber"
    :next-steps="nextSteps"
    :show-contact="true"
    :primary-action="{ label: 'View Request', action: navigateToRequest }"
    :secondary-action="{ label: 'Download PDF', action: downloadPDF }"
    close-label="Create Another Request"
    @close="closeSuccessModal"
  />
</template>

<script setup lang="ts">
import { useWarehousingRequestStore } from './submit-warehousing-request-store'
import SuccessModal from '~/components/ui-library/modals/SuccessModal.vue'

const store = useWarehousingRequestStore()

const nextSteps = [
  'Our warehousing team will assess space availability within 1 business hour',
  'You will receive a detailed storage quote and terms within 3 hours',
  'Upon approval, we will coordinate cargo receipt and storage setup',
  'You can monitor your inventory status through the dashboard'
]

const navigateToRequest = () => {
  store.resetForm()
  navigateTo('/dashboard/requests/warehousing')
}

const closeSuccessModal = () => {
  store.resetForm()
  navigateTo('/dashboard/requests/warehousing')
}

const downloadPDF = async () => {
  if (process.server) return

  try {
    const { generateWarehousingRequestPDF } = await import('~/lib/pdf/warehousingRequestPdfGenerator')

    const formData = {
      storageType: store.form.storageType || '',
      securityLevel: store.form.securityLevel || 'STANDARD',
      estimatedVolume: store.form.estimatedVolume || 0,
      estimatedWeight: store.form.estimatedWeight || 0,
      estimatedStorageDuration: store.form.estimatedStorageDuration || { value: 1, unit: 'months' },
      plannedStartDate: store.form.plannedStartDate || new Date().toISOString(),
      plannedEndDate: store.form.plannedEndDate || undefined,
      handlingServices: store.form.handlingServices || [],
      valueAddedServices: store.form.valueAddedServices || [],
      requiresTemperatureControl: store.form.requiresTemperatureControl || false,
      requiresHumidityControl: store.form.requiresHumidityControl || false,
      requiresSpecialHandling: store.form.requiresSpecialHandling || false,
      specialInstructions: store.form.specialInstructions || undefined,
      billingType: store.form.billingType || 'MONTHLY',
      cargo: {
        description: store.form.cargo?.description || '',
        cargoType: store.form.cargo?.cargoType || 'GENERAL_CARGO',
        packaging: store.form.cargo?.packaging || 'PALLETS',
        quantity: store.form.cargo?.quantity || 0,
        unitType: store.form.cargo?.unitType || '',
        value: store.form.cargo?.value || 0,
        currency: store.form.cargo?.currency || 'EUR'
      },
      priority: store.form.priority || 'NORMAL'
    }

    await generateWarehousingRequestPDF(formData, {
      requestNumber: store.generatedRequestNumber,
      createdAt: new Date()
    })
  } catch (error) {
    console.error('Error generating PDF:', error)
    alert(`Error generating PDF: ${error instanceof Error ? error.message : 'Unknown error'}`)
  }
}
</script>
