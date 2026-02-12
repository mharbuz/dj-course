import type { PdfContext } from './PdfContext'

export interface IPdfSection {
  render(ctx: PdfContext): number
}
