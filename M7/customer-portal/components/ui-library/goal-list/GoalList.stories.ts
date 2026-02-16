import { ref } from 'vue'
import type { Meta, StoryObj } from '@storybook/vue3'
import GoalList from './GoalList.vue'
import type { Goal } from './GoalList.vue'
import { ClipboardDocumentCheckIcon } from '@heroicons/vue/24/outline'

const meta: Meta<typeof GoalList> = {
  title: 'UI Library/GoalList',
  component: GoalList,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof meta>

const sampleGoals: Goal[] = [
  { label: '30min Morning Yoga', completed: true },
  { label: '10k Steps', completed: false },
  { label: 'Drink 2L Water', completed: true }
]

export const Default: Story = {
  args: {
    title: "Today's Goals",
    goals: sampleGoals
  }
}

export const WithAddButton: Story = {
  args: {
    title: "Today's Goals",
    goals: sampleGoals,
    showAddButton: true
  }
}

export const ServiceRequestsContext: Story = {
  args: {
    title: "Today's Goals",
    goals: [
      { label: 'Submit transportation request', completed: true },
      { label: 'Review pending warehousing quote', completed: false },
      { label: 'Track shipment status', completed: true }
    ],
    icon: ClipboardDocumentCheckIcon,
    showAddButton: true
  }
}

export const Interactive: Story = {
  render: (args) => ({
    components: { GoalList },
    setup() {
      const goals = ref([
        { label: 'Submit transportation request', completed: true },
        { label: 'Review pending warehousing quote', completed: false },
        { label: 'Track shipment status', completed: true }
      ])
      const onToggle = (index: number) => {
        goals.value[index] = { ...goals.value[index], completed: !goals.value[index].completed }
      }
      return { args, goals, onToggle }
    },
    template: `
      <GoalList
        :title="args.title"
        :goals="goals"
        :icon="args.icon"
        :show-add-button="args.showAddButton"
        @toggle="onToggle"
      />
    `
  }),
  args: {
    title: "Today's Goals",
    icon: ClipboardDocumentCheckIcon,
    showAddButton: true
  }
}
