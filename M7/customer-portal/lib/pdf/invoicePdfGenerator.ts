import { PdfDocumentBuilder } from './builder/PdfDocumentBuilder'
import { KeyValueSection } from './sections/KeyValueSection'
import { formatCurrency, formatDate } from './builder/constants'

interface InvoiceData {
  id: string
  number: string
  description: string
  date: Date
  amount: number
  status: 'Paid' | 'Unpaid' | 'Overdue'
  dueDate: Date
}

export async function generateInvoicePDF(invoice: InvoiceData): Promise<void> {
  const builder = await PdfDocumentBuilder.create({
    title: 'Invoice',
    filename: `Invoice_${invoice.number}.pdf`
  })

  builder
    .addHeader()
    .addSection(
      new KeyValueSection('Invoice Details', [
        { label: 'Invoice Number', value: invoice.number, fullWidth: true },
        { label: 'Invoice ID', value: invoice.id, fullWidth: true },
        { label: 'Description', value: invoice.description, fullWidth: true },
        { label: 'Amount', value: formatCurrency(invoice.amount, 'USD'), fullWidth: true },
        { label: 'Status', value: invoice.status, fullWidth: true },
        { label: 'Invoice Date', value: formatDate(invoice.date), fullWidth: true },
        { label: 'Due Date', value: formatDate(invoice.dueDate), fullWidth: true }
      ])
    )

  await builder.build()
}
