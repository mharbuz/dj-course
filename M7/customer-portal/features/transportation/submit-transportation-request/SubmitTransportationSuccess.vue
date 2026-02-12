<template>
  <SuccessModal
    :show="store.success"
    title="Transportation Request Submitted Successfully!"
    message="Your transportation request has been received and is being processed by our logistics team."
    :reference-number="store.requestNumber"
    :next-steps="nextSteps"
    :show-contact="true"
    :primary-action="{ label: 'View Request', action: navigateToRequest }"
    :secondary-action="{ label: 'Download PDF', action: downloadPDF }"
    close-label="Create Another Request"
    @close="closeSuccessModal"
  />
</template>

<script setup lang="ts">
import { useTransportationRequestStore } from './submit-transportation-request-store'
import SuccessModal from '~/components/ui-library/modals/SuccessModal.vue'

const store = useTransportationRequestStore()

const nextSteps = [
  'Our team will review your request within 2 business hours',
  'You will receive a detailed quote via email within 4 hours',
  'Once approved, we will schedule pickup and provide tracking information',
  'You can track your shipment progress in real-time through the dashboard'
]

const navigateToRequest = () => {
  store.resetForm()
  navigateTo('/dashboard/requests/transportation')
}

const closeSuccessModal = () => {
  store.resetForm()
  navigateTo('/dashboard/requests/transportation')
}

const downloadPDF = async () => {
  if (process.server) return

  try {
    const { generateTransportationRequestPDF } = await import('~/lib/pdf/transportationRequestPdfGenerator')
    await generateTransportationRequestPDF(store.form, {
      requestNumber: store.requestNumber,
      createdAt: new Date()
    })
  } catch (error) {
    console.error('Error generating PDF:', error)
    alert('Error generating PDF. Please try again.')
  }
}
</script>
