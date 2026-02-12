<template>
  <div>
    <WarehousingDetailsHeader
      :request-id="requestId"
      :request="request"
      :pdf-loading="pdfLoading"
      @download-pdf="downloadPDF"
    />

    <div v-if="isLoading" class="card p-8 text-center">
      <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
      <p class="text-gray-500 dark:text-gray-400">Loading request details...</p>
    </div>

    <div v-else-if="isError" class="card p-8 text-center">
      <ExclamationTriangleIcon class="h-8 w-8 text-red-600 dark:text-red-400 mx-auto mb-4" />
      <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-2">Error Loading Request</h3>
      <button @click="() => refetch()" class="btn-primary">Try Again</button>
    </div>

    <div v-else-if="request" class="space-y-8">
      <WarehousingDetailsOverview :request="request" />
      <WarehousingDetailsStorageRequirements :request="request" />
      <WarehousingDetailsCargo :request="request" />
      <WarehousingDetailsSpecialRequirements :request="request" />
      <WarehousingDetailsServices :request="request" />
      <WarehousingDetailsTimeline :request="request" />
      <WarehousingDetailsPricing :request="request" />
    </div>

    <div v-else-if="!isLoading" class="card p-8 text-center">
      <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-2">Request Not Found</h3>
      <NuxtLink to="/dashboard/requests" class="btn-primary">Back to Requests</NuxtLink>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ExclamationTriangleIcon } from '@heroicons/vue/24/outline'
import { useWarehousingRequestDetails } from './warehousing-request-details-api'
import WarehousingDetailsHeader from './WarehousingDetailsHeader.vue'
import WarehousingDetailsOverview from './WarehousingDetailsOverview.vue'
import WarehousingDetailsStorageRequirements from './WarehousingDetailsStorageRequirements.vue'
import WarehousingDetailsCargo from './WarehousingDetailsCargo.vue'
import WarehousingDetailsSpecialRequirements from './WarehousingDetailsSpecialRequirements.vue'
import WarehousingDetailsServices from './WarehousingDetailsServices.vue'
import WarehousingDetailsTimeline from './WarehousingDetailsTimeline.vue'
import WarehousingDetailsPricing from './WarehousingDetailsPricing.vue'

const route = useRoute()
const requestId = route.params.id as string
const pdfLoading = ref(false)

const { data: request, isLoading, isError, refetch } = useWarehousingRequestDetails(requestId)

const downloadPDF = async () => {
  if (!request.value || process.server) return
  pdfLoading.value = true
  try {
    const { PDFGenerator } = await import('~/lib/pdf/pdfGenerator')
    await new Promise(resolve => setTimeout(resolve, 500))
    await PDFGenerator.generateWarehousingRequestPDF(request.value)
  } catch (error) {
    console.error('Error generating PDF:', error)
    alert('Error generating PDF. Please try again.')
  } finally {
    pdfLoading.value = false
  }
}
</script>
