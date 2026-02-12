import jsPDF from 'jspdf'
import { loadLogo, LAYOUT } from './constants'
import { createPdfContext, type PdfContext } from './PdfContext'
import type { IPdfSection } from './types'
import { FooterSection } from '../sections/FooterSection'
import { HeaderSection } from '../sections/HeaderSection'

export interface PdfDocumentBuilderConfig {
  title: string
  filename: string
}

export class PdfDocumentBuilder {
  private doc: jsPDF
  private ctx: PdfContext
  private config: PdfDocumentBuilderConfig
  private sections: IPdfSection[] = []
  private logoDataUrl: string | null = null

  private constructor(config: PdfDocumentBuilderConfig) {
    this.doc = new jsPDF()
    this.ctx = createPdfContext(this.doc)
    this.config = config
  }

  static async create(config: PdfDocumentBuilderConfig): Promise<PdfDocumentBuilder> {
    const builder = new PdfDocumentBuilder(config)
    builder.logoDataUrl = await loadLogo()
    return builder
  }

  addHeader(): this {
    this.sections.push(new HeaderSection(this.config.title, this.logoDataUrl))
    return this
  }

  addSection(section: IPdfSection): this {
    this.sections.push(section)
    return this
  }

  async build(): Promise<void> {
    for (const section of this.sections) {
      this.ctx.currentY = section.render(this.ctx)
    }

    const footerSection = new FooterSection()
    footerSection.render(this.ctx)

    this.doc.save(this.config.filename)
  }

  getDoc(): jsPDF {
    return this.doc
  }

  getContext(): PdfContext {
    return this.ctx
  }
}

export function ensureSpace(ctx: PdfContext, minHeight: number): void {
  if (ctx.currentY + minHeight > ctx.pageHeight - LAYOUT.minBottomMargin) {
    ctx.doc.addPage()
    ctx.currentY = 20
  }
}
