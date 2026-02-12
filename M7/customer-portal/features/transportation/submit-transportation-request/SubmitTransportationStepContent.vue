<template>
  <div class="card p-6 mb-8">
    <h2 class="text-xl font-bold text-gray-900 dark:text-white mb-2">
      {{ store.stepTitle }}
    </h2>
    <p class="text-sm text-gray-500 dark:text-gray-400 mb-6">
      {{ store.stepDescription }}
    </p>

    <StepServiceType v-if="store.currentStep === 1" />
    <StepPickupInfo v-if="store.currentStep === 2" />
    <StepDeliveryInfo v-if="store.currentStep === 3" />
    <StepCargoInfo v-if="store.currentStep === 4" />
    <StepSpecialInstructions v-if="store.currentStep === 5" />
    <StepReview v-if="store.currentStep === 6" />
  </div>
</template>

<script setup lang="ts">
import { useTransportationRequestStore } from './submit-transportation-request-store'
import StepServiceType from './StepServiceType.vue'
import StepPickupInfo from './StepPickupInfo.vue'
import StepDeliveryInfo from './StepDeliveryInfo.vue'
import StepCargoInfo from './StepCargoInfo.vue'
import StepSpecialInstructions from './StepSpecialInstructions.vue'
import StepReview from './StepReview.vue'

const store = useTransportationRequestStore()
</script>
