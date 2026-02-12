<template>
  <div class="card p-6">
    <div class="flex items-center justify-between mb-6">
      <h2 class="text-lg font-medium text-gray-900 dark:text-white">
        Request Overview
      </h2>
      <span
        :class="[
          'inline-flex items-center px-3 py-1 rounded-full text-sm font-medium',
          statusColor
        ]"
      >
        {{ formatEnum(request.status) }}
      </span>
    </div>

    <div class="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
      <div>
        <dt class="text-sm font-medium text-gray-500 dark:text-gray-400">Request Number</dt>
        <dd class="mt-1 text-sm text-gray-900 dark:text-white">{{ request.requestNumber }}</dd>
      </div>
      <div>
        <dt class="text-sm font-medium text-gray-500 dark:text-gray-400">Storage Type</dt>
        <dd class="mt-1 text-sm text-gray-900 dark:text-white">{{ formatEnum(request.storageType) }}</dd>
      </div>
      <div>
        <dt class="text-sm font-medium text-gray-500 dark:text-gray-400">Priority</dt>
        <dd class="mt-1 text-sm text-gray-900 dark:text-white">{{ formatEnum(request.priority) }}</dd>
      </div>
      <div>
        <dt class="text-sm font-medium text-gray-500 dark:text-gray-400">Storage Location</dt>
        <dd class="mt-1 text-sm text-gray-900 dark:text-white">{{ request.storageLocation || 'Not assigned' }}</dd>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { WarehousingRequest } from './warehousing-request-details.model'

const props = defineProps<{ request: WarehousingRequest }>()

const formatEnum = (value: string) =>
  value.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, l => l.toUpperCase())

const statusColors: Record<string, string> = {
  'SUBMITTED': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
  'APPROVED': 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
  'RECEIVED': 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200',
  'STORED': 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
  'COMPLETED': 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
}

const statusColor = computed(() =>
  statusColors[props.request.status] || 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
)
</script>
