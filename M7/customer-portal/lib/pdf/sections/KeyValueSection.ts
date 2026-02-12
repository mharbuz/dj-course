import type { PdfContext } from '../builder/PdfContext'
import { ensureSpace } from '../builder/PdfDocumentBuilder'
import { LAYOUT } from '../builder/constants'

export interface KeyValueRow {
  label: string
  value: string
  fullWidth?: boolean
}

export class KeyValueSection {
  constructor(
    private sectionTitle: string,
    private rows: KeyValueRow[]
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

    ctx.doc.setFontSize(10)
    const valueColumnWidth = ctx.contentWidth - (LAYOUT.valueColumnX - ctx.margin)

    for (const row of this.rows) {
      ensureSpace(ctx, LAYOUT.rowHeight + 4)

      ctx.doc.setFont('helvetica', 'bold')
      ctx.doc.text(`${row.label}:`, LAYOUT.labelColumnX, ctx.currentY)
      ctx.doc.setFont('helvetica', 'normal')

      const textWidth = row.fullWidth ? ctx.contentWidth : valueColumnWidth
      const lines = ctx.doc.splitTextToSize(row.value, textWidth)
      const valueX = row.fullWidth ? LAYOUT.labelColumnX : LAYOUT.valueColumnX
      ctx.doc.text(lines, valueX, ctx.currentY + 4)
      ctx.currentY += Math.max(lines.length * 4, LAYOUT.rowHeight) + 2
    }

    ctx.currentY += 7
    return ctx.currentY
  }
}
