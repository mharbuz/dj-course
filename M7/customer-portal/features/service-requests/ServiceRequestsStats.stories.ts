import type { Meta, StoryObj } from '@storybook/vue3'
import ServiceRequestsStats from './ServiceRequestsStats.vue'

const meta: Meta<typeof ServiceRequestsStats> = {
  title: 'Features/Service Requests/ServiceRequestsStats',
  component: ServiceRequestsStats,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {}
