<template>
  <div class="card p-6">
    <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-4">
      Value-Added Services
    </h3>

    <div class="grid grid-cols-2 gap-4 sm:grid-cols-3">
      <label
        v-for="service in valueAddedServices"
        :key="service.value"
        class="flex items-center"
      >
        <input
          v-model="store.form.valueAddedServices"
          type="checkbox"
          :value="service.value"
          class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
        />
        <span class="ml-2 text-sm text-gray-700 dark:text-gray-300">{{ service.name }}</span>
      </label>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useWarehousingRequestStore } from './submit-warehousing-request-store'

const store = useWarehousingRequestStore()

const valueAddedServices = [
  { value: 'LABELING', name: 'Labeling' },
  { value: 'REPACKAGING', name: 'Repackaging' },
  { value: 'QUALITY_CONTROL', name: 'Quality Control' },
  { value: 'CROSS_DOCKING', name: 'Cross Docking' },
  { value: 'KITTING', name: 'Kitting' }
]
</script>
