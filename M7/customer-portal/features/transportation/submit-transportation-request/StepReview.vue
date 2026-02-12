<template>
  <div class="space-y-6">
    <!-- Service Type -->
    <div class="border-b border-gray-200 dark:border-gray-700 pb-4">
      <div class="flex justify-between items-center">
        <h3 class="text-lg font-medium text-gray-900 dark:text-white">Service Type</h3>
        <button
          @click="store.goToStep(1)"
          class="text-sm text-primary-600 hover:text-primary-500 dark:text-primary-400"
        >
          Edit
        </button>
      </div>
      <p class="mt-1 text-sm text-gray-600 dark:text-gray-400">
        {{ getServiceTypeName(store.form.serviceType) }}
      </p>
    </div>

    <!-- Pickup Information -->
    <div class="border-b border-gray-200 dark:border-gray-700 pb-4">
      <div class="flex justify-between items-center">
        <h3 class="text-lg font-medium text-gray-900 dark:text-white">Pickup Information</h3>
        <button
          @click="store.goToStep(2)"
          class="text-sm text-primary-600 hover:text-primary-500 dark:text-primary-400"
        >
          Edit
        </button>
      </div>
      <div class="mt-2 text-sm text-gray-600 dark:text-gray-400">
        <p>{{ store.form.pickupLocation.address.street }}</p>
        <p>{{ store.form.pickupLocation.address.city }}, {{ store.form.pickupLocation.address.country }}</p>
        <p class="mt-2">Contact: {{ store.form.pickupLocation.contactPerson }}</p>
        <p>Phone: {{ store.form.pickupLocation.contactPhone }}</p>
        <p class="mt-2">Pickup Date: {{ formatDate(store.form.requestedPickupDate) }}</p>
      </div>
    </div>

    <!-- Delivery Information -->
    <div class="border-b border-gray-200 dark:border-gray-700 pb-4">
      <div class="flex justify-between items-center">
        <h3 class="text-lg font-medium text-gray-900 dark:text-white">Delivery Information</h3>
        <button
          @click="store.goToStep(3)"
          class="text-sm text-primary-600 hover:text-primary-500 dark:text-primary-400"
        >
          Edit
        </button>
      </div>
      <div class="mt-2 text-sm text-gray-600 dark:text-gray-400">
        <p>{{ store.form.deliveryLocation.address.street }}</p>
        <p>{{ store.form.deliveryLocation.address.city }}, {{ store.form.deliveryLocation.address.country }}</p>
        <p class="mt-2">Contact: {{ store.form.deliveryLocation.contactPerson }}</p>
        <p>Phone: {{ store.form.deliveryLocation.contactPhone }}</p>
        <p v-if="store.form.requestedDeliveryDate" class="mt-2">
          Delivery Date: {{ formatDate(store.form.requestedDeliveryDate) }}
        </p>
      </div>
    </div>

    <!-- Cargo Information -->
    <div class="border-b border-gray-200 dark:border-gray-700 pb-4">
      <div class="flex justify-between items-center">
        <h3 class="text-lg font-medium text-gray-900 dark:text-white">Cargo Information</h3>
        <button
          @click="store.goToStep(4)"
          class="text-sm text-primary-600 hover:text-primary-500 dark:text-primary-400"
        >
          Edit
        </button>
      </div>
      <div class="mt-2 text-sm text-gray-600 dark:text-gray-400">
        <p>{{ store.form.cargo.description }}</p>
        <p class="mt-2">Weight: {{ store.form.cargo.weight }} kg</p>
        <p>Type: {{ formatEnum(store.form.cargo.cargoType) }}</p>
        <p>Packaging: {{ formatEnum(store.form.cargo.packaging) }}</p>
        <p>Quantity: {{ store.form.cargo.quantity }} {{ store.form.cargo.unitType }}</p>
        <div class="mt-2 flex space-x-4">
          <span v-if="store.form.cargo.fragile" class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200">
            Fragile
          </span>
          <span v-if="store.form.requiresInsurance" class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
            Insured
          </span>
        </div>
      </div>
    </div>

    <!-- Special Instructions -->
    <div>
      <div class="flex justify-between items-center">
        <h3 class="text-lg font-medium text-gray-900 dark:text-white">Special Instructions</h3>
        <button
          @click="store.goToStep(5)"
          class="text-sm text-primary-600 hover:text-primary-500 dark:text-primary-400"
        >
          Edit
        </button>
      </div>
      <div class="mt-2 text-sm text-gray-600 dark:text-gray-400">
        <p v-if="store.form.specialInstructions">{{ store.form.specialInstructions }}</p>
        <p v-else>No special instructions provided.</p>
        <p class="mt-2">Priority: {{ formatEnum(store.form.priority) }}</p>
        <p v-if="store.form.requiresCustomsClearance" class="mt-2">
          Requires customs clearance
        </p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useTransportationRequestStore } from './submit-transportation-request-store'

const store = useTransportationRequestStore()

const serviceTypes: Record<string, string> = {
  'FULL_TRUCKLOAD': 'Full Truckload (FTL)',
  'LESS_THAN_TRUCKLOAD': 'Less Than Truckload (LTL)',
  'EXPRESS_DELIVERY': 'Express Delivery',
  'OVERSIZED_CARGO': 'Oversized Cargo',
  'HAZARDOUS_MATERIALS': 'Hazardous Materials'
}

const getServiceTypeName = (type: string) => serviceTypes[type] || type

const formatDate = (dateString: string) => {
  if (!dateString) return 'Not specified'
  return new Date(dateString).toLocaleDateString('en-US', {
    year: 'numeric', month: 'long', day: 'numeric'
  })
}

const formatEnum = (value: string) =>
  value.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, l => l.toUpperCase())
</script>
