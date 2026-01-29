/**
 * Ollama configuration validation using Zod
 */

import { z } from 'zod';

/**
 * Ollama configuration schema
 */
export const OllamaConfigSchema = z.object({
  engine: z.literal('OLLAMA').default('OLLAMA'),
  baseUrl: z.string().url().default('http://localhost:11434'),
  modelName: z.string().min(1, 'Model name is required'),
  temperature: z.number().min(0).max(2).optional(),
  topP: z.number().min(0).max(1).optional(),
  topK: z.number().int().min(1).optional(),
});

export type OllamaConfig = z.infer<typeof OllamaConfigSchema>;

/**
 * Validate and parse Ollama configuration from environment
 */
export function validateOllamaConfig(): OllamaConfig {
  const config = {
    engine: 'OLLAMA' as const,
    baseUrl: process.env.OLLAMA_BASE_URL || 'http://localhost:11434',
    modelName: process.env.OLLAMA_MODEL_NAME || 'gpt-oss',
    temperature: process.env.TEMPERATURE ? parseFloat(process.env.TEMPERATURE) : undefined,
    topP: process.env.TOP_P ? parseFloat(process.env.TOP_P) : undefined,
    topK: process.env.TOP_K ? parseInt(process.env.TOP_K, 10) : undefined,
  };

  return OllamaConfigSchema.parse(config);
}
