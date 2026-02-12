import type jsPDF from 'jspdf'

export interface PdfContext {
  doc: jsPDF
  pageHeight: number
  pageWidth: number
  currentY: number
  margin: number
  contentWidth: number
}

export function createPdfContext(doc: jsPDF): PdfContext {
  const pageHeight = doc.internal.pageSize.height
  const pageWidth = doc.internal.pageSize.width
  return {
    doc,
    pageHeight,
    pageWidth,
    currentY: 55,
    margin: 20,
    contentWidth: pageWidth - 40
  }
}
