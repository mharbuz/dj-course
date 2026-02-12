import { MongoClient, Db } from 'mongodb';

const MONGO_URI = process.env.MONGO_URI!;

const client = new MongoClient(MONGO_URI);
let db: Db;

export async function connectDb(): Promise<Db> {
  await client.connect();
  db = client.db('invoicesdb');
  console.log('Connected to MongoDB');
  return db;
}

export function getDb(): Db {
  return db;
}

export async function closeDb(): Promise<void> {
  await client.close();
}
