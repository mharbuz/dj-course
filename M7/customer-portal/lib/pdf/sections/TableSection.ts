import type { PdfContext } from '../builder/PdfContext'
import { ensureSpace } from '../builder/PdfDocumentBuilder'
import { LAYOUT } from '../builder/constants'

export interface TableColumn<T = Record<string, unknown>> {
  header: string
  width: number
  key: keyof T | string
  format?: (value: unknown) => string
}

export class TableSection<T extends Record<string, unknown> = Record<string, unknown>> {
  constructor(
    private sectionTitle: string,
    private columns: TableColumn<T>[],
    private rows: T[]
  ) {}

  render(ctx: PdfContext): number {
    ensureSpace(ctx, LAYOUT.sectionHeaderHeight + LAYOUT.sectionSpacing)

    ctx.doc.setFontSize(14)
    ctx.doc.setFont('helvetica', 'bold')
    ctx.doc.setFillColor(248, 250, 252)
    ctx.doc.rect(
      ctx.margin,
      ctx.currentY,
      ctx.contentWidth,
      LAYOUT.sectionHeaderHeight,
      'F'
    )
    ctx.doc.text(this.sectionTitle, ctx.margin + 2, ctx.currentY + 5.5)
    ctx.currentY += LAYOUT.sectionSpacing

    const colWidths = this.columns.map((c) => c.width)

    ensureSpace(ctx, 10)
    ctx.doc.setFontSize(9)
    ctx.doc.setFont('helvetica', 'bold')
    ctx.doc.setFillColor(240, 240, 240)
    ctx.doc.rect(ctx.margin, ctx.currentY, ctx.contentWidth, 6, 'F')

    let xPos = ctx.margin + 2
    this.columns.forEach((col, idx) => {
      ctx.doc.text(col.header, xPos, ctx.currentY + 4)
      xPos += colWidths[idx]
    })
    ctx.currentY += 10

    ctx.doc.setFontSize(8)
    ctx.doc.setFont('helvetica', 'normal')

    for (const row of this.rows) {
      ensureSpace(ctx, 10)
      xPos = ctx.margin + 2
      let maxLines = 1

      this.columns.forEach((col, idx) => {
        const value = row[col.key as keyof T]
        const formatted = col.format ? col.format(value) : String(value ?? '')
        const cellText = ctx.doc.splitTextToSize(formatted, colWidths[idx] - 4)
        ctx.doc.text(cellText, xPos, ctx.currentY + 4)
        maxLines = Math.max(maxLines, cellText.length)
        xPos += colWidths[idx]
      })

      ctx.currentY += maxLines * 4 + 2
      ctx.doc.setDrawColor(200, 200, 200)
      ctx.doc.line(ctx.margin, ctx.currentY - 1, ctx.pageWidth - ctx.margin, ctx.currentY - 1)
    }

    ctx.currentY += 7
    return ctx.currentY
  }
}
