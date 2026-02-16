<template>
  <div class="grid grid-cols-1 gap-6 lg:grid-cols-2">
    <!-- Today's Progress -->
    <div class="card p-6">
      <div class="mb-6 flex items-center gap-2">
        <ChartBarIcon class="h-6 w-6 text-gray-500 dark:text-gray-400" />
        <div>
          <h2 class="text-lg font-semibold text-gray-900 dark:text-white">
            {{ stats.progressTitle }}
          </h2>
          <p class="text-sm text-gray-500 dark:text-gray-400">
            {{ stats.progressSubtitle }}
          </p>
        </div>
      </div>
      <div class="flex flex-wrap justify-around gap-6">
        <CircularProgress
          v-for="(metric, index) in stats.metrics"
          :key="index"
          :value="metric.value"
          :unit="metric.unit"
          :label="metric.label"
          :percentage="metric.percentage"
          :color="metric.color as 'red' | 'green' | 'blue'"
        />
      </div>
    </div>

    <!-- Today's Goals -->
    <div class="card p-6">
      <GoalList
        :title="stats.goalsTitle"
        :goals="goals"
        :icon="ClipboardDocumentCheckIcon"
        :show-add-button="true"
        @add="onAddGoal"
        @toggle="onToggleGoal"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { ChartBarIcon, ClipboardDocumentCheckIcon } from '@heroicons/vue/24/outline'
import CircularProgress from '~/components/ui-library/circular-progress/CircularProgress.vue'
import GoalList from '~/components/ui-library/goal-list/GoalList.vue'
import { mockServiceRequestsStats } from './service-requests-stats.mocks'
import type { ServiceRequestGoal } from './service-requests-stats.model'

const stats = mockServiceRequestsStats
const goals = ref<ServiceRequestGoal[]>([...mockServiceRequestsStats.goals])

const onAddGoal = () => {
  // Placeholder for add goal action
  console.log('Add goal clicked')
}

const onToggleGoal = (index: number) => {
  goals.value[index] = { ...goals.value[index], completed: !goals.value[index].completed }
}
</script>
