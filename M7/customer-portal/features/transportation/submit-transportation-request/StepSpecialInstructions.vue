<template>
  <div>
    <div>
      <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
        Special Instructions
      </label>
      <textarea
        v-model="store.form.specialInstructions"
        rows="4"
        class="input"
        placeholder="Any special handling requirements, delivery instructions, or additional information..."
      ></textarea>
    </div>

    <div class="mt-6">
      <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
        Priority
      </label>
      <div class="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <label
          v-for="priority in priorities"
          :key="priority.value"
          :class="[
            'relative flex cursor-pointer rounded-lg border p-4 focus:outline-none',
            store.form.priority === priority.value
              ? 'border-primary-600 ring-2 ring-primary-600 bg-primary-50 dark:bg-primary-900/20'
              : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
          ]"
        >
          <input
            v-model="store.form.priority"
            type="radio"
            :value="priority.value"
            class="sr-only"
          />
          <div class="flex items-center">
            <div class="text-sm">
              <div class="font-medium text-gray-900 dark:text-white">
                {{ priority.name }}
              </div>
            </div>
          </div>
        </label>
      </div>
    </div>

    <div class="mt-6">
      <label class="flex items-center">
        <input
          v-model="store.form.requiresCustomsClearance"
          type="checkbox"
          class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
        />
        <span class="ml-2 text-sm text-gray-700 dark:text-gray-300">Requires Customs Clearance</span>
      </label>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useTransportationRequestStore } from './submit-transportation-request-store'

const store = useTransportationRequestStore()

const priorities = [
  { value: 'LOW', name: 'Low' },
  { value: 'NORMAL', name: 'Normal' },
  { value: 'HIGH', name: 'High' },
  { value: 'URGENT', name: 'Urgent' }
]
</script>
