<template>
  <div class="flex flex-col items-center">
    <div class="relative">
      <svg
        class="transform -rotate-90"
        :width="size"
        :height="size"
        viewBox="0 0 100 100"
      >
        <!-- Background circle -->
        <circle
          cx="50"
          cy="50"
          r="42"
          fill="none"
          class="stroke-gray-200 dark:stroke-gray-700"
          stroke-width="8"
        />
        <!-- Progress circle -->
        <circle
          cx="50"
          cy="50"
          r="42"
          fill="none"
          :stroke="progressColor"
          stroke-width="8"
          stroke-linecap="round"
          :stroke-dasharray="circumference"
          :stroke-dashoffset="offset"
          class="transition-all duration-500 ease-out"
        />
      </svg>
      <div
        class="absolute inset-0 flex flex-col items-center justify-center"
      >
        <span class="text-2xl font-bold text-gray-900 dark:text-white">
          {{ value }}
        </span>
        <span class="text-sm text-gray-500 dark:text-gray-400">
          {{ unit }}
        </span>
      </div>
    </div>
    <div class="mt-2 text-center">
      <p class="text-sm font-medium text-gray-700 dark:text-gray-300">
        {{ label }}
      </p>
      <p class="text-xs text-gray-500 dark:text-gray-400">
        {{ percentage }}%
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
export type CircularProgressColor = 'red' | 'green' | 'blue'

interface Props {
  value: string | number
  unit: string
  label: string
  percentage: number
  color?: CircularProgressColor
  size?: number
}

const props = withDefaults(defineProps<Props>(), {
  color: 'red',
  size: 120
})

const circumference = 2 * Math.PI * 42

const offset = computed(() => {
  const progress = Math.min(100, Math.max(0, props.percentage)) / 100
  return circumference * (1 - progress)
})

const progressColor = computed(() => {
  const colors = {
    red: 'rgb(239 68 68)',    // red-500
    green: 'rgb(34 197 94)',   // green-500
    blue: 'rgb(59 130 246)'    // blue-500
  }
  return colors[props.color]
})
</script>
