<template>
  <div class="card p-6">
    <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-4">
      Special Instructions
    </h3>
    <textarea
      v-model="store.form.specialInstructions"
      rows="4"
      class="input"
      placeholder="Any special storage requirements, handling instructions, or additional information..."
    ></textarea>
  </div>
</template>

<script setup lang="ts">
import { useWarehousingRequestStore } from './submit-warehousing-request-store'

const store = useWarehousingRequestStore()
</script>
