import { createClient } from 'redis';

const REDIS_URL = process.env.REDIS_URL!;

const redisClient = createClient({ url: REDIS_URL });

redisClient.on('error', (err) => console.error('Redis error:', err));

export async function connectRedis() {
  await redisClient.connect();
  console.log('Connected to Redis');
}

export async function closeRedis() {
  await redisClient.quit();
}

export { redisClient };
