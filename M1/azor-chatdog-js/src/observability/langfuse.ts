/**
 * Langfuse observability singleton
 *
 * Lazy-initializes a Langfuse client from environment variables.
 * If credentials are missing, all operations are no-ops — the app
 * works without Langfuse.
 */

import { Langfuse } from 'langfuse';

let instance: Langfuse | null = null;
let initialized = false;

/**
 * Get the Langfuse client instance.
 * Returns null if env vars are missing (graceful no-op).
 */
export function getLangfuse(): Langfuse | null {
  if (initialized) return instance;
  initialized = true;

  const secretKey = process.env.LANGFUSE_SECRET_KEY;
  const publicKey = process.env.LANGFUSE_PUBLIC_KEY;
  const baseUrl = process.env.LANGFUSE_BASE_URL;

  if (!secretKey || !publicKey) {
    return null;
  }

  instance = new Langfuse({
    secretKey,
    publicKey,
    baseUrl,
  });

  return instance;
}

/**
 * Flush pending events and shut down the Langfuse client.
 * Safe to call even if Langfuse was never initialized.
 */
export async function shutdownLangfuse(): Promise<void> {
  if (instance) {
    await instance.shutdownAsync();
    instance = null;
    initialized = false;
  }
}
