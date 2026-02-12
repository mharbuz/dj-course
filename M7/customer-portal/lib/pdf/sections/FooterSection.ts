import type { PdfContext } from '../builder/PdfContext'
import { FOOTER_LINES, LAYOUT } from '../builder/constants'

export class FooterSection {
  render(ctx: PdfContext): number {
    ctx.doc.setDrawColor(200, 200, 200)
    ctx.doc.line(
      ctx.margin,
      ctx.pageHeight - LAYOUT.footerTopMargin,
      ctx.pageWidth - ctx.margin,
      ctx.pageHeight - LAYOUT.footerTopMargin
    )
    ctx.doc.setFontSize(8)
    ctx.doc.setTextColor(100, 100, 100)

    FOOTER_LINES.forEach((line, idx) => {
      ctx.doc.text(line, ctx.margin, ctx.pageHeight - 18 + idx * 6)
    })

    const pageCount = ctx.doc.getNumberOfPages()
    for (let i = 1; i <= pageCount; i++) {
      ctx.doc.setPage(i)
      ctx.doc.text(
        `Page ${i} of ${pageCount}`,
        ctx.pageWidth - 30,
        ctx.pageHeight - 12
      )
    }

    return ctx.currentY
  }
}
