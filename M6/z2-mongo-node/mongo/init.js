db = db.getSiblingDB('invoicesdb');

db.createCollection('invoices');

db.invoices.insertMany([
  {
    invoiceNumber: 'INV-001',
    customer: 'Acme Corp',
    items: [
      { description: 'Widget A', quantity: 10, unitPrice: 25.00 },
      { description: 'Widget B', quantity: 5, unitPrice: 50.00 }
    ],
    totalAmount: 500.00,
    status: 'paid',
    issuedAt: new Date('2025-01-15'),
    dueDate: new Date('2025-02-15')
  },
  {
    invoiceNumber: 'INV-002',
    customer: 'Globex Inc',
    items: [
      { description: 'Service Plan', quantity: 1, unitPrice: 1200.00 }
    ],
    totalAmount: 1200.00,
    status: 'pending',
    issuedAt: new Date('2025-02-01'),
    dueDate: new Date('2025-03-01')
  },
  {
    invoiceNumber: 'INV-003',
    customer: 'Initech LLC',
    items: [
      { description: 'Consulting Hours', quantity: 40, unitPrice: 150.00 },
      { description: 'Travel Expenses', quantity: 1, unitPrice: 350.00 }
    ],
    totalAmount: 6350.00,
    status: 'overdue',
    issuedAt: new Date('2024-12-01'),
    dueDate: new Date('2025-01-01')
  }
]);

print('Seed data inserted: 3 invoices');
