export type ProgressMetricColor = 'red' | 'green' | 'blue'

export interface ProgressMetric {
  value: string | number
  unit: string
  label: string
  percentage: number
  color: ProgressMetricColor
}

export interface ServiceRequestGoal {
  label: string
  completed: boolean
}

export interface ServiceRequestsStats {
  progressTitle: string
  progressSubtitle: string
  metrics: ProgressMetric[]
  goalsTitle: string
  goals: ServiceRequestGoal[]
}
