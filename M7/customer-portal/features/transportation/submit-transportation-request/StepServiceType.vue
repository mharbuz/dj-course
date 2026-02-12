<template>
  <div>
    <div class="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      <label
        v-for="service in serviceTypes"
        :key="service.value"
        :class="[
          'relative flex cursor-pointer rounded-lg border p-4 focus:outline-none',
          store.form.serviceType === service.value
            ? 'border-primary-600 ring-2 ring-primary-600 bg-primary-50 dark:bg-primary-900/20'
            : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500',
          validationErrors['serviceType'] && !store.form.serviceType ? 'border-red-500' : ''
        ]"
      >
        <input
          v-model="store.form.serviceType"
          type="radio"
          :value="service.value"
          class="sr-only"
          required
        />
        <div class="flex items-center">
          <div class="text-sm">
            <div class="font-medium text-gray-900 dark:text-white">
              {{ service.name }}
            </div>
            <div class="text-gray-500 dark:text-gray-400">
              {{ service.description }}
            </div>
          </div>
        </div>
      </label>
    </div>
    <div v-if="validationErrors['serviceType']" class="text-red-500 text-sm mt-2">
      {{ validationErrors['serviceType'] }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useTransportationRequestStore } from './submit-transportation-request-store'

const store = useTransportationRequestStore()
const validationErrors = computed(() => store.validationErrors)

const serviceTypes = [
  { value: 'FULL_TRUCKLOAD', name: 'Full Truckload (FTL)', description: 'Dedicated truck for your cargo' },
  { value: 'LESS_THAN_TRUCKLOAD', name: 'Less Than Truckload (LTL)', description: 'Shared truck space' },
  { value: 'EXPRESS_DELIVERY', name: 'Express Delivery', description: 'Priority fast delivery' },
  { value: 'OVERSIZED_CARGO', name: 'Oversized Cargo', description: 'Special handling for large items' },
  { value: 'HAZARDOUS_MATERIALS', name: 'Hazardous Materials', description: 'ADR compliant transport' }
]
</script>
