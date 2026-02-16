import type { ServiceRequestsStats } from './service-requests-stats.model'

export const mockServiceRequestsStats: ServiceRequestsStats = {
  progressTitle: "Today's Progress",
  progressSubtitle: 'Activity',
  metrics: [
    {
      value: '12',
      unit: 'requests',
      label: 'Transportation',
      percentage: 85,
      color: 'red'
    },
    {
      value: '8',
      unit: 'requests',
      label: 'Warehousing',
      percentage: 70,
      color: 'green'
    },
    {
      value: '3',
      unit: 'active',
      label: 'In Progress',
      percentage: 83,
      color: 'blue'
    }
  ],
  goalsTitle: "Today's Goals",
  goals: [
    { label: 'Submit transportation request', completed: true },
    { label: 'Review pending warehousing quote', completed: false },
    { label: 'Track shipment status', completed: true }
  ]
}
