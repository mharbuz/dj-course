<template>
  <div class="grid grid-cols-1 gap-6 sm:grid-cols-2">
    <div class="sm:col-span-2">
      <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
        Delivery Address *
      </label>
      <input
        v-model="store.form.deliveryLocation.address.street"
        type="text"
        required
        :class="['input', validationErrors['deliveryLocation.address.street'] ? 'border-red-500 focus:ring-red-500 focus:border-red-500' : '']"
        placeholder="Enter delivery address"
      />
      <div v-if="validationErrors['deliveryLocation.address.street']" class="text-red-500 text-sm mt-1">
        {{ validationErrors['deliveryLocation.address.street'] }}
      </div>
    </div>

    <div>
      <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
        City *
      </label>
      <input
        v-model="store.form.deliveryLocation.address.city"
        type="text"
        required
        :class="['input', validationErrors['deliveryLocation.address.city'] ? 'border-red-500 focus:ring-red-500 focus:border-red-500' : '']"
        placeholder="Enter city"
      />
      <div v-if="validationErrors['deliveryLocation.address.city']" class="text-red-500 text-sm mt-1">
        {{ validationErrors['deliveryLocation.address.city'] }}
      </div>
    </div>

    <div>
      <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
        Country *
      </label>
      <select
        v-model="store.form.deliveryLocation.address.country"
        required
        :class="['input', validationErrors['deliveryLocation.address.country'] ? 'border-red-500 focus:ring-red-500 focus:border-red-500' : '']"
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
      <div v-if="validationErrors['deliveryLocation.address.country']" class="text-red-500 text-sm mt-1">
        {{ validationErrors['deliveryLocation.address.country'] }}
      </div>
    </div>

    <div>
      <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
        Contact Person *
      </label>
      <input
        v-model="store.form.deliveryLocation.contactPerson"
        type="text"
        required
        :class="['input', validationErrors['deliveryLocation.contactPerson'] ? 'border-red-500 focus:ring-red-500 focus:border-red-500' : '']"
        placeholder="Contact person name"
      />
      <div v-if="validationErrors['deliveryLocation.contactPerson']" class="text-red-500 text-sm mt-1">
        {{ validationErrors['deliveryLocation.contactPerson'] }}
      </div>
    </div>

    <div>
      <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
        Contact Phone *
      </label>
      <input
        v-model="store.form.deliveryLocation.contactPhone"
        type="tel"
        required
        :class="['input', validationErrors['deliveryLocation.contactPhone'] ? 'border-red-500 focus:ring-red-500 focus:border-red-500' : '']"
        placeholder="Phone number"
      />
      <div v-if="validationErrors['deliveryLocation.contactPhone']" class="text-red-500 text-sm mt-1">
        {{ validationErrors['deliveryLocation.contactPhone'] }}
      </div>
    </div>

    <div>
      <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
        Preferred Delivery Date
      </label>
      <input
        v-model="store.form.requestedDeliveryDate"
        type="date"
        class="input"
      />
    </div>

    <div>
      <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
        Unloading Type
      </label>
      <select
        v-model="store.form.deliveryLocation.loadingType"
        class="input"
      >
        <option value="DOCK">Dock Unloading</option>
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
