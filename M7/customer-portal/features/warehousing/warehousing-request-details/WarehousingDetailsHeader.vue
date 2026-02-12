<template>
  <div class="mb-8 flex items-center justify-between">
    <div>
      <div class="flex items-center space-x-4 mb-2">
        <NuxtLink
          to="/dashboard/requests"
          class="text-success-600 hover:text-success-500 dark:text-success-400 flex items-center"
        >
          <ArrowLeftIcon class="w-5 h-5 mr-1" />
          Back to Requests
        </NuxtLink>
      </div>
      <h1 class="text-2xl font-bold text-gray-900 dark:text-white">
        Warehousing Request {{ requestId }}
      </h1>
      <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
        Detailed view of your warehousing request
      </p>
    </div>
    <div class="flex space-x-3">
      <button class="btn-outline">
        <CubeIcon class="w-5 h-5 mr-2" />
        View Inventory
      </button>
      <button
        v-if="request"
        @click="$emit('download-pdf')"
        :disabled="pdfLoading"
        class="btn-primary"
      >
        <DocumentArrowDownIcon class="w-5 h-5 mr-2" />
        <span v-if="!pdfLoading">Download PDF</span>
        <span v-else>Generating PDF...</span>
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ArrowLeftIcon, CubeIcon, DocumentArrowDownIcon } from '@heroicons/vue/24/outline'
import type { WarehousingRequest } from './warehousing-request-details.model'

defineProps<{
  requestId: string
  request: WarehousingRequest | null | undefined
  pdfLoading: boolean
}>()

defineEmits<{
  'download-pdf': []
}>()
</script>
