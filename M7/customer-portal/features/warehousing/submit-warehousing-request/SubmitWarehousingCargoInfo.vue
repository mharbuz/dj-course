<template>
  <div class="card p-6">
    <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-4">
      Cargo Information
    </h3>

    <div class="grid grid-cols-1 gap-6 sm:grid-cols-2">
      <div class="sm:col-span-2">
        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Cargo Description *
        </label>
        <textarea
          v-model="store.form.cargo.description"
          rows="3"
          required
          class="input"
          placeholder="Describe the items to be stored"
        ></textarea>
      </div>

      <div>
        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Cargo Type
        </label>
        <select v-model="store.form.cargo.cargoType" class="input">
          <option value="GENERAL_CARGO">General Cargo</option>
          <option value="PERISHABLE">Perishable</option>
          <option value="HAZARDOUS">Hazardous</option>
          <option value="VALUABLE">Valuable</option>
          <option value="BULK_DRY">Bulk Dry</option>
          <option value="BULK_LIQUID">Bulk Liquid</option>
        </select>
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
          Unit Type
        </label>
        <input
          v-model="store.form.cargo.unitType"
          type="text"
          class="input"
          placeholder="e.g., pallets, boxes, pieces"
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
  </div>
</template>

<script setup lang="ts">
import { useWarehousingRequestStore } from './submit-warehousing-request-store'

const store = useWarehousingRequestStore()
</script>
