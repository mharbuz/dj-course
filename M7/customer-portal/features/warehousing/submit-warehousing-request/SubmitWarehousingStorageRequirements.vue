<template>
  <div class="card p-6">
    <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-4">
      Storage Requirements
    </h3>

    <div class="grid grid-cols-1 gap-6 sm:grid-cols-2">
      <div>
        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Storage Type *
        </label>
        <select v-model="store.form.storageType" required class="input">
          <option value="">Select storage type</option>
          <option value="AMBIENT">Ambient (15-25°C)</option>
          <option value="PERISHABLE">Perishable (2-8°C)</option>
          <option value="FROZEN">Frozen (-18°C)</option>
          <option value="DRY">Dry (15-25°C)</option>
          <option value="HAZARDOUS">Hazardous Materials</option>
          <option value="SECURE">High Security</option>
        </select>
      </div>

      <div>
        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Security Level
        </label>
        <select v-model="store.form.securityLevel" class="input">
          <option value="STANDARD">Standard</option>
          <option value="HIGH">High Security</option>
          <option value="MAXIMUM">Maximum Security</option>
        </select>
      </div>

      <div>
        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Estimated Volume (m³) *
        </label>
        <input
          v-model.number="store.form.estimatedVolume"
          type="number"
          min="0"
          step="0.1"
          required
          class="input"
          placeholder="Enter volume in cubic meters"
        />
      </div>

      <div>
        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Estimated Weight (kg) *
        </label>
        <input
          v-model.number="store.form.estimatedWeight"
          type="number"
          min="0"
          required
          class="input"
          placeholder="Enter weight in kg"
        />
      </div>

      <div>
        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Storage Duration *
        </label>
        <div class="flex space-x-2">
          <input
            v-model.number="store.form.estimatedStorageDuration.value"
            type="number"
            min="1"
            required
            class="input flex-1"
            placeholder="Duration"
          />
          <select
            v-model="store.form.estimatedStorageDuration.unit"
            required
            class="input w-32"
          >
            <option value="days">Days</option>
            <option value="weeks">Weeks</option>
            <option value="months">Months</option>
            <option value="years">Years</option>
          </select>
        </div>
      </div>

      <div>
        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Planned Start Date *
        </label>
        <input
          v-model="store.form.plannedStartDate"
          type="date"
          required
          class="input"
        />
      </div>
    </div>

    <div class="mt-6 space-y-4">
      <div class="flex items-center space-x-6">
        <label class="flex items-center">
          <input
            v-model="store.form.requiresTemperatureControl"
            type="checkbox"
            class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
          />
          <span class="ml-2 text-sm text-gray-700 dark:text-gray-300">Requires Temperature Control</span>
        </label>

        <label class="flex items-center">
          <input
            v-model="store.form.requiresHumidityControl"
            type="checkbox"
            class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
          />
          <span class="ml-2 text-sm text-gray-700 dark:text-gray-300">Requires Humidity Control</span>
        </label>

        <label class="flex items-center">
          <input
            v-model="store.form.requiresSpecialHandling"
            type="checkbox"
            class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
          />
          <span class="ml-2 text-sm text-gray-700 dark:text-gray-300">Requires Special Handling</span>
        </label>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useWarehousingRequestStore } from './submit-warehousing-request-store'

const store = useWarehousingRequestStore()
</script>
