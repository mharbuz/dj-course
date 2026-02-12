import { defineStore } from 'pinia'
import type { SubmitWarehousingRequestForm } from './submit-warehousing-request.model'
import { submitWarehousingRequest } from './submit-warehousing-request-api'

export const useWarehousingRequestStore = defineStore('warehousingRequest', {
  state: () => ({
    loading: false,
    showSuccessModal: false,
    generatedRequestNumber: '',
    form: {
      storageType: '',
      estimatedVolume: 0,
      estimatedWeight: 0,
      estimatedStorageDuration: { value: 1, unit: 'months' as const },
      plannedStartDate: '',
      plannedEndDate: '',
      handlingServices: [] as string[],
      valueAddedServices: [] as string[],
      securityLevel: 'STANDARD',
      requiresTemperatureControl: false,
      requiresHumidityControl: false,
      requiresSpecialHandling: false,
      specialInstructions: '',
      billingType: 'MONTHLY',
      cargo: {
        description: '',
        cargoType: 'GENERAL_CARGO',
        packaging: 'PALLETS',
        quantity: 1,
        unitType: 'pallets',
        value: 0,
        currency: 'EUR'
      },
      priority: 'NORMAL'
    } as SubmitWarehousingRequestForm
  }),

  actions: {
    async submitRequest() {
      this.loading = true
      try {
        const response = await submitWarehousingRequest(this.form)
        this.generatedRequestNumber = response.requestNumber
        this.showSuccessModal = true
      } catch (error) {
        console.error('Request submission failed:', error)
      } finally {
        this.loading = false
      }
    },

    resetForm() {
      this.showSuccessModal = false
      this.generatedRequestNumber = ''
      this.form = {
        storageType: '',
        estimatedVolume: 0,
        estimatedWeight: 0,
        estimatedStorageDuration: { value: 1, unit: 'months' },
        plannedStartDate: '',
        plannedEndDate: '',
        handlingServices: [],
        valueAddedServices: [],
        securityLevel: 'STANDARD',
        requiresTemperatureControl: false,
        requiresHumidityControl: false,
        requiresSpecialHandling: false,
        specialInstructions: '',
        billingType: 'MONTHLY',
        cargo: {
          description: '',
          cargoType: 'GENERAL_CARGO',
          packaging: 'PALLETS',
          quantity: 1,
          unitType: 'pallets',
          value: 0,
          currency: 'EUR'
        },
        priority: 'NORMAL'
      }
    }
  }
})
