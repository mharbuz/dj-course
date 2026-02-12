<template>
  <div>
    <SubmitTransportationHeader />
    <SubmitTransportationTimeline />
    <SubmitTransportationStepContent />
    <SubmitTransportationNavigation />
    <SubmitTransportationSuccess />
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { useTransportationRequestStore } from './submit-transportation-request-store'
import SubmitTransportationHeader from './SubmitTransportationHeader.vue'
import SubmitTransportationTimeline from './SubmitTransportationTimeline.vue'
import SubmitTransportationStepContent from './SubmitTransportationStepContent.vue'
import SubmitTransportationNavigation from './SubmitTransportationNavigation.vue'
import SubmitTransportationSuccess from './SubmitTransportationSuccess.vue'

const store = useTransportationRequestStore()
onMounted(() => store.resetForm())
</script>
