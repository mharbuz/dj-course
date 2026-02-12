<template>
  <div>
    <div class="grid grid-cols-1 gap-6 sm:grid-cols-2">
      <div class="sm:col-span-2">
        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Cargo Description *
        </label>
        <textarea
          v-model="store.form.cargo.description"
          rows="3"
          required
          :class="['input', validationErrors['cargo.description'] ? 'border-red-500 focus:ring-red-500 focus:border-red-500' : '']"
          placeholder="Describe the cargo to be transported"
        ></textarea>
        <div v-if="validationErrors['cargo.description']" class="text-red-500 text-sm mt-1">
          {{ validationErrors['cargo.description'] }}
        </div>
      </div>

      <div>
        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Cargo Type
        </label>
        <select v-model="store.form.cargo.cargoType" class="input">
          <option value="GENERAL_CARGO">General Cargo</option>
          <option value="PERISHABLE">Perishable</option>
          <option value="HAZARDOUS">Hazardous</option>
          <option value="OVERSIZED">Oversized</option>
          <option value="VALUABLE">Valuable</option>
        </select>
      </div>

      <div>
        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Weight (kg) *
        </label>
        <input
          v-model.number="store.form.cargo.weight"
          type="number"
          min="0"
          required
          :class="['input', validationErrors['cargo.weight'] ? 'border-red-500 focus:ring-red-500 focus:border-red-500' : '']"
          placeholder="Enter weight in kg"
        />
        <div v-if="validationErrors['cargo.weight']" class="text-red-500 text-sm mt-1">
          {{ validationErrors['cargo.weight'] }}
        </div>
      </div>

      <div>
        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Packaging Type
        </label>
        <select v-model="store.form.cargo.packaging" class="input">
          <option value="PALLETS">Pallets</option>
          <option value="BOXES">Boxes</option>
          <option value="CRATES">Crates</option>
          <option value="BULK">Bulk</option>
          <option value="CONTAINERS">Containers</option>
        </select>
      </div>

      <div>
        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Quantity
        </label>
        <input
          v-model.number="store.form.cargo.quantity"
          type="number"
          min="1"
          class="input"
          placeholder="Number of units"
        />
      </div>

      <div>
        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Estimated Value (EUR)
        </label>
        <input
          v-model.number="store.form.cargo.value"
          type="number"
          min="0"
          class="input"
          placeholder="Enter estimated value"
        />
      </div>
    </div>

    <div class="mt-6 space-y-4">
      <div class="flex items-center space-x-6">
        <label class="flex items-center">
          <input
            v-model="store.form.cargo.fragile"
            type="checkbox"
            class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
          />
          <span class="ml-2 text-sm text-gray-700 dark:text-gray-300">Fragile</span>
        </label>

        <label class="flex items-center">
          <input
            v-model="store.form.cargo.stackable"
            type="checkbox"
            class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
          />
          <span class="ml-2 text-sm text-gray-700 dark:text-gray-300">Stackable</span>
        </label>

        <label class="flex items-center">
          <input
            v-model="store.form.requiresInsurance"
            type="checkbox"
            class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
          />
          <span class="ml-2 text-sm text-gray-700 dark:text-gray-300">Requires Insurance</span>
        </label>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useTransportationRequestStore } from './submit-transportation-request-store'

const store = useTransportationRequestStore()
const validationErrors = computed(() => store.validationErrors)
</script>
