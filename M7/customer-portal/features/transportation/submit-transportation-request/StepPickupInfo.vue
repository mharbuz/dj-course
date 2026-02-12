<template>
  <div class="grid grid-cols-1 gap-6 sm:grid-cols-2">
    <div class="sm:col-span-2">
      <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
        Pickup Address *
      </label>
      <input
        v-model="store.form.pickupLocation.address.street"
        type="text"
        required
        :class="['input', validationErrors['pickupLocation.address.street'] ? 'border-red-500 focus:ring-red-500 focus:border-red-500' : '']"
        placeholder="Enter pickup address"
      />
      <div v-if="validationErrors['pickupLocation.address.street']" class="text-red-500 text-sm mt-1">
        {{ validationErrors['pickupLocation.address.street'] }}
      </div>
    </div>

    <div>
      <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
        City *
      </label>
      <input
        v-model="store.form.pickupLocation.address.city"
        type="text"
        required
        :class="['input', validationErrors['pickupLocation.address.city'] ? 'border-red-500 focus:ring-red-500 focus:border-red-500' : '']"
        placeholder="Enter city"
      />
      <div v-if="validationErrors['pickupLocation.address.city']" class="text-red-500 text-sm mt-1">
        {{ validationErrors['pickupLocation.address.city'] }}
      </div>
    </div>

    <div>
      <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
        Country *
      </label>
      <select
        v-model="store.form.pickupLocation.address.country"
        required
        :class="['input', validationErrors['pickupLocation.address.country'] ? 'border-red-500 focus:ring-red-500 focus:border-red-500' : '']"
      >
        <option value="">Select country</option>
        <option value="Poland">Poland</option>
        <option value="Germany">Germany</option>
        <option value="Czech Republic">Czech Republic</option>
        <option value="Slovakia">Slovakia</option>
        <option value="Hungary">Hungary</option>
        <option value="Austria">Austria</option>
        <option value="Netherlands">Netherlands</option>
        <option value="Belgium">Belgium</option>
        <option value="France">France</option>
      </select>
      <div v-if="validationErrors['pickupLocation.address.country']" class="text-red-500 text-sm mt-1">
        {{ validationErrors['pickupLocation.address.country'] }}
      </div>
    </div>

    <div>
      <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
        Contact Person *
      </label>
      <input
        v-model="store.form.pickupLocation.contactPerson"
        type="text"
        required
        :class="['input', validationErrors['pickupLocation.contactPerson'] ? 'border-red-500 focus:ring-red-500 focus:border-red-500' : '']"
        placeholder="Contact person name"
      />
      <div v-if="validationErrors['pickupLocation.contactPerson']" class="text-red-500 text-sm mt-1">
        {{ validationErrors['pickupLocation.contactPerson'] }}
      </div>
    </div>

    <div>
      <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
        Contact Phone *
      </label>
      <input
        v-model="store.form.pickupLocation.contactPhone"
        type="tel"
        required
        :class="['input', validationErrors['pickupLocation.contactPhone'] ? 'border-red-500 focus:ring-red-500 focus:border-red-500' : '']"
        placeholder="Phone number"
      />
      <div v-if="validationErrors['pickupLocation.contactPhone']" class="text-red-500 text-sm mt-1">
        {{ validationErrors['pickupLocation.contactPhone'] }}
      </div>
    </div>

    <div>
      <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
        Preferred Pickup Date *
      </label>
      <input
        v-model="store.form.requestedPickupDate"
        type="date"
        required
        :class="['input', validationErrors['requestedPickupDate'] ? 'border-red-500 focus:ring-red-500 focus:border-red-500' : '']"
      />
      <div v-if="validationErrors['requestedPickupDate']" class="text-red-500 text-sm mt-1">
        {{ validationErrors['requestedPickupDate'] }}
      </div>
    </div>

    <div>
      <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
        Loading Type
      </label>
      <select
        v-model="store.form.pickupLocation.loadingType"
        class="input"
      >
        <option value="DOCK">Dock Loading</option>
        <option value="GROUND">Ground Level</option>
        <option value="CRANE">Crane Required</option>
        <option value="FORKLIFT">Forklift Available</option>
      </select>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useTransportationRequestStore } from './submit-transportation-request-store'

const store = useTransportationRequestStore()
const validationErrors = computed(() => store.validationErrors)
</script>
