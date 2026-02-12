<template>
  <div class="flex justify-between">
    <button
      v-if="!store.isFirstStep"
      @click="store.prevStep"
      class="btn-outline"
    >
      Back
    </button>
    <div v-else></div>

    <div>
      <button
        v-if="!store.isLastStep"
        @click="handleNextStep"
        :disabled="!store.isStepValid"
        :class="[
          'btn-primary',
          !store.isStepValid ? 'opacity-50 cursor-not-allowed' : ''
        ]"
      >
        Next
      </button>

      <button
        v-else
        @click="handleSubmit"
        :disabled="store.loading"
        class="btn-primary"
      >
        <span v-if="!store.loading">Submit Request</span>
        <span v-else class="flex items-center">
          <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          Submitting...
        </span>
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useQueryClient } from '@tanstack/vue-query'
import { useTransportationRequestStore } from './submit-transportation-request-store'

const store = useTransportationRequestStore()
const queryClient = useQueryClient()

const handleNextStep = () => {
  if (store.validateCurrentStep()) {
    store.nextStep()
  }
}

const handleSubmit = async () => {
  await store.submitRequest()

  if (store.success) {
    await queryClient.invalidateQueries({
      queryKey: ['transportationRequests'],
      exact: false
    })
    await queryClient.refetchQueries({
      queryKey: ['transportationRequests'],
      exact: false
    })
  }
}
</script>
