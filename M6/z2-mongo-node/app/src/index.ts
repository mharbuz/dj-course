import express, { Request, Response } from 'express';
import { connectDb, getDb, closeDb } from './database';
import { connectRedis, closeRedis, redisClient } from './redis';

const CACHE_KEY = 'invoices';
const CACHE_TTL = 60; // seconds

const app = express();
app.use(express.json());

const port = process.env.PORT || 3000;

// Health check
app.get('/', (_req: Request, res: Response) => {
  res.json({ status: 'ok', timestamp: new Date() });
});

// GET /invoices — read from cache, fallback to MongoDB
app.get('/invoices', async (_req: Request, res: Response) => {
  try {
    const cached = await redisClient.get(CACHE_KEY);

    if (cached) {
      console.log('Cache HIT — returning invoices from Redis');
      res.json(JSON.parse(cached));
      return;
    }

    console.log('Cache MISS — fetching invoices from MongoDB');
    const invoices = await getDb().collection('invoices').find().toArray();

    await redisClient.set(CACHE_KEY, JSON.stringify(invoices), { EX: CACHE_TTL });

    res.json(invoices);
  } catch (err) {
    console.error('GET /invoices error:', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// POST /invoices — insert into MongoDB, invalidate cache
app.post('/invoices', async (req: Request, res: Response) => {
  try {
    const invoice = req.body;

    if (!invoice.invoiceNumber || !invoice.customer) {
      res.status(400).json({ error: 'invoiceNumber and customer are required' });
      return;
    }

    const result = await getDb().collection('invoices').insertOne({
      ...invoice,
      createdAt: new Date()
    });

    // Invalidate cache so next GET fetches fresh data
    await redisClient.del(CACHE_KEY);
    console.log('Cache invalidated after POST');

    res.status(201).json({ insertedId: result.insertedId, ...invoice });
  } catch (err) {
    console.error('POST /invoices error:', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Start server after connecting to databases
async function main() {
  await connectDb();
  await connectRedis();

  app.listen(port, () => {
    console.log(`Server running on port ${port}`);
  });
}

// Graceful shutdown
process.on('SIGINT', async () => {
  console.log('Shutting down...');
  await closeRedis();
  await closeDb();
  process.exit(0);
});

main().catch((err) => {
  console.error('Failed to start:', err);
  process.exit(1);
});
