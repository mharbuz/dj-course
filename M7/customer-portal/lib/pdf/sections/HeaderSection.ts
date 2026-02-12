import type { PdfContext } from '../builder/PdfContext'

export class HeaderSection {
  constructor(
    private title: string,
    private logoDataUrl: string | null
  ) {}

  render(ctx: PdfContext): number {
    if (this.logoDataUrl) {
      ctx.doc.addImage(this.logoDataUrl, 'PNG', 15, 15, 15, 15)
    }

    ctx.doc.setFontSize(16)
    ctx.doc.setFont('helvetica', 'bold')
    ctx.doc.text(this.title, 20, 35)

    ctx.doc.setFontSize(10)
    ctx.doc.setFont('helvetica', 'normal')
    ctx.doc.text('Deliveroo Logistics', 20, 42)

    return 55
  }
}
