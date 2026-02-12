<template>
  <div class="grid grid-cols-1 gap-8 lg:grid-cols-2">
    <!-- Handling Services -->
    <div class="card p-6">
      <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-4">
        Handling Services
      </h3>
      <div class="space-y-2">
        <div
          v-for="service in request.handlingServices"
          :key="service"
          class="flex items-center"
        >
          <CheckIcon class="w-4 h-4 text-green-500 mr-2" />
          <span class="text-sm text-gray-900 dark:text-white">{{ formatEnum(service) }}</span>
        </div>
        <div v-if="request.handlingServices.length === 0" class="text-sm text-gray-500 dark:text-gray-400">
          No additional handling services requested
        </div>
      </div>
    </div>

    <!-- Value-Added Services -->
    <div class="card p-6">
      <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-4">
        Value-Added Services
      </h3>
      <div class="space-y-2">
        <div
          v-for="service in request.valueAddedServices"
          :key="service"
          class="flex items-center"
        >
          <CheckIcon class="w-4 h-4 text-green-500 mr-2" />
          <span class="text-sm text-gray-900 dark:text-white">{{ formatEnum(service) }}</span>
        </div>
        <div v-if="request.valueAddedServices.length === 0" class="text-sm text-gray-500 dark:text-gray-400">
          No value-added services requested
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { CheckIcon } from '@heroicons/vue/24/outline'
import type { WarehousingRequest } from './warehousing-request-details.model'

defineProps<{ request: WarehousingRequest }>()

const formatEnum = (value: string) =>
  value.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, l => l.toUpperCase())
</script>
