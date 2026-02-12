import { PdfDocumentBuilder } from './builder/PdfDocumentBuilder'
import { KeyValueSection } from './sections/KeyValueSection'
import { TableSection } from './sections/TableSection'
import { formatCurrency, formatDate } from './builder/constants'

interface MetricsData {
  totalShipments: number
  onTimeDelivery: number
  totalCost: number
  storageVolume: number
}

interface RoutePerformanceData {
  route: string
  shipments: number
  onTimePercentage: number
  avgCost: number
  totalRevenue: number
}

interface ReportsData {
  dateRange: { from: string; to: string }
  metrics: MetricsData
  routePerformance: RoutePerformanceData[]
}

export async function generateReportsPDF(reportsData: ReportsData): Promise<void> {
  const fromDateStr = reportsData.dateRange.from.replace(/-/g, '')
  const toDateStr = reportsData.dateRange.to.replace(/-/g, '')
  const periodText = `${formatDate(reportsData.dateRange.from)} - ${formatDate(reportsData.dateRange.to)}`

  const builder = await PdfDocumentBuilder.create({
    title: 'Logistics Report',
    filename: `Logistics_Report_${fromDateStr}_${toDateStr}.pdf`
  })

  const formatEur = (v: unknown) =>
    new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'EUR',
      maximumFractionDigits: 0
    }).format(Number(v))

  builder
    .addHeader()
    .addSection(
      new KeyValueSection('Report Period', [{ label: 'Period', value: periodText }])
    )
    .addSection(
      new KeyValueSection('Key Metrics', [
        { label: 'Total Shipments', value: String(reportsData.metrics.totalShipments) },
        { label: 'On-Time Delivery', value: `${reportsData.metrics.onTimeDelivery.toFixed(1)}%` },
        { label: 'Total Cost', value: formatCurrency(reportsData.metrics.totalCost, 'EUR') },
        {
          label: 'Storage Volume',
          value: `${reportsData.metrics.storageVolume.toLocaleString()} m³`
        }
      ])
    )
    .addSection(
      new TableSection<RoutePerformanceData>(
        'Route Performance',
        [
          { header: 'Route', width: 60, key: 'route' },
          { header: 'Shipments', width: 30, key: 'shipments' },
          {
            header: 'On-Time %',
            width: 30,
            key: 'onTimePercentage',
            format: (v) => `${v}%`
          },
          { header: 'Avg Cost', width: 30, key: 'avgCost', format: formatEur },
          { header: 'Revenue', width: 30, key: 'totalRevenue', format: formatEur }
        ],
        reportsData.routePerformance
      )
    )

  await builder.build()
}
