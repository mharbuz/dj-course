import type { Meta, StoryObj } from '@storybook/vue3'
import CircularProgress from './CircularProgress.vue'
import type { CircularProgressColor } from './CircularProgress.vue'

const meta: Meta<typeof CircularProgress> = {
  title: 'UI Library/CircularProgress',
  component: CircularProgress,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  argTypes: {
    color: {
      control: { type: 'select' },
      options: ['red', 'green', 'blue'] as CircularProgressColor[]
    }
  }
}

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    value: '420',
    unit: 'cal',
    label: 'Move',
    percentage: 85,
    color: 'red'
  }
}

export const AllColors: Story = {
  render: () => ({
    components: { CircularProgress },
    template: `
      <div class="flex gap-12">
        <CircularProgress value="420" unit="cal" label="Move" :percentage="85" color="red" />
        <CircularProgress value="35" unit="min" label="Exercise" :percentage="70" color="green" />
        <CircularProgress value="10" unit="hrs" label="Stand" :percentage="83" color="blue" />
      </div>
    `
  })
}

export const ServiceRequestsContext: Story = {
  render: () => ({
    components: { CircularProgress },
    template: `
      <div class="flex gap-12">
        <CircularProgress value="12" unit="requests" label="Transportation" :percentage="85" color="red" />
        <CircularProgress value="8" unit="requests" label="Warehousing" :percentage="70" color="green" />
        <CircularProgress value="3" unit="active" label="In Progress" :percentage="83" color="blue" />
      </div>
    `
  })
}
