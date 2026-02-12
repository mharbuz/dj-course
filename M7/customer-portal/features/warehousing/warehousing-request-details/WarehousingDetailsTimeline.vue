<template>
  <div class="card p-6">
    <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-6">
      Request Timeline
    </h3>
    <Timeline :items="request.progressUpdates" prefix="at" />
  </div>
</template>

<script setup lang="ts">
import type { WarehousingRequest } from './warehousing-request-details.model'
import Timeline from '~/components/ui-library/timeline/Timeline.vue'

defineProps<{ request: WarehousingRequest }>()
</script>
