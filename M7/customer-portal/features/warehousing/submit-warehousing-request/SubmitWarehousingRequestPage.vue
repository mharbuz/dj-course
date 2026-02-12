<template>
  <div>
    <SubmitWarehousingHeader />
    <form @submit.prevent="handleSubmit" class="space-y-8">
      <SubmitWarehousingStorageRequirements />
      <SubmitWarehousingCargoInfo />
      <SubmitWarehousingHandlingServices />
      <SubmitWarehousingValueAddedServices />
      <SubmitWarehousingSpecialInstructions />
      <SubmitWarehousingFormActions />
    </form>
    <SubmitWarehousingSuccess />
  </div>
</template>

<script setup lang="ts">
import { useQueryClient } from '@tanstack/vue-query'
import { useWarehousingRequestStore } from './submit-warehousing-request-store'
import SubmitWarehousingHeader from './SubmitWarehousingHeader.vue'
import SubmitWarehousingStorageRequirements from './SubmitWarehousingStorageRequirements.vue'
import SubmitWarehousingCargoInfo from './SubmitWarehousingCargoInfo.vue'
import SubmitWarehousingHandlingServices from './SubmitWarehousingHandlingServices.vue'
import SubmitWarehousingValueAddedServices from './SubmitWarehousingValueAddedServices.vue'
import SubmitWarehousingSpecialInstructions from './SubmitWarehousingSpecialInstructions.vue'
import SubmitWarehousingFormActions from './SubmitWarehousingFormActions.vue'
import SubmitWarehousingSuccess from './SubmitWarehousingSuccess.vue'

const store = useWarehousingRequestStore()
const queryClient = useQueryClient()

const handleSubmit = async () => {
  await store.submitRequest()
  if (store.showSuccessModal) {
    await queryClient.invalidateQueries({ queryKey: ['warehousingRequests'], exact: false })
    await queryClient.refetchQueries({ queryKey: ['warehousingRequests'], exact: false })
  }
}
</script>
