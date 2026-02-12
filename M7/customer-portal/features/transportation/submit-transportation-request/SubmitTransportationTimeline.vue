<template>
  <div class="mb-8">
    <ShipmentTimeline
      :shipment-data="timelineData"
      @status-change="handleStepChange"
      :disabled-steps="disabledSteps"
    />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useTransportationRequestStore } from './submit-transportation-request-store'
import ShipmentTimeline from '~/features/transportation/shipment-details/ShipmentTimeline.vue'
import type { ShipmentTimelineData } from '~/features/transportation/shipment-details/shipment-timeline.model'

const store = useTransportationRequestStore()

const timelineData = computed<ShipmentTimelineData>(() => ({
  id: 'new-request',
  trackingId: 'new-request',
  currentStatusIndex: store.currentStep - 1,
  statuses: [
    { id: 'service-type', name: 'Service Type', timestamp: '', icon: 'truck', completed: store.currentStep > 1 },
    { id: 'pickup-info', name: 'Pickup', timestamp: '', icon: 'package', completed: store.currentStep > 2 },
    { id: 'delivery-info', name: 'Delivery', timestamp: '', icon: 'truck', completed: store.currentStep > 3 },
    { id: 'cargo-info', name: 'Cargo', timestamp: '', icon: 'box', completed: store.currentStep > 4 },
    { id: 'special-instructions', name: 'Instructions', timestamp: '', icon: 'package', completed: store.currentStep > 5 },
    { id: 'review', name: 'Review', timestamp: '', icon: 'check', completed: store.currentStep > 6 }
  ]
}))

const disabledSteps = computed(() => {
  const disabled = []
  for (let i = 1; i <= store.totalSteps; i++) {
    if (!store.canAccessStep(i)) {
      disabled.push(i - 1)
    }
  }
  return disabled
})

const handleStepChange = (stepIndex: number) => {
  const step = stepIndex + 1
  if (store.canAccessStep(step)) {
    store.goToStep(step)
  }
}
</script>
