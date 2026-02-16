<template>
  <div>
    <div class="mb-4 flex items-center justify-between">
      <div class="flex items-center gap-2">
        <component
          v-if="icon"
          :is="icon"
          class="h-5 w-5 text-gray-500 dark:text-gray-400"
        />
        <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
          {{ title }}
        </h3>
      </div>
      <button
        v-if="showAddButton"
        type="button"
        class="rounded-full p-2 text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700 dark:text-gray-400"
        aria-label="Add goal"
        @click="$emit('add')"
      >
        <PlusIcon class="h-5 w-5" />
      </button>
    </div>
    <ul class="space-y-3">
      <li
        v-for="(goal, index) in goals"
        :key="index"
        class="card flex cursor-pointer items-center gap-3 rounded-lg border px-4 py-3 transition-colors hover:bg-gray-50 dark:hover:bg-gray-700/50"
        role="button"
        tabindex="0"
        @click="$emit('toggle', index)"
        @keydown.enter.prevent="$emit('toggle', index)"
        @keydown.space.prevent="$emit('toggle', index)"
      >
        <div
          :class="[
            'flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full',
            goal.completed
              ? 'bg-green-100 dark:bg-green-900'
              : 'bg-gray-100 dark:bg-gray-700'
          ]"
        >
          <CheckCircleIcon
            v-if="goal.completed"
            class="h-5 w-5 text-green-600 dark:text-green-400"
          />
          <span
            v-else
            class="h-5 w-5 rounded-full border-2 border-gray-400 dark:border-gray-500"
          />
        </div>
        <span
          :class="[
            'flex-1 text-sm font-medium',
            goal.completed
              ? 'text-gray-900 dark:text-white line-through'
              : 'text-gray-500 dark:text-gray-400'
          ]"
        >
          {{ goal.label }}
        </span>
      </li>
    </ul>
  </div>
</template>

<script setup lang="ts">
import { PlusIcon } from '@heroicons/vue/24/outline'
import { CheckCircleIcon } from '@heroicons/vue/24/solid'

export interface Goal {
  label: string
  completed: boolean
}

interface Props {
  title: string
  goals: Goal[]
  icon?: object
  showAddButton?: boolean
}

withDefaults(defineProps<Props>(), {
  showAddButton: false
})

defineEmits<{
  add: []
  toggle: [index: number]
}>()
</script>
