/**
 * Gemini configuration validation using Zod
 */

import { z } from 'zod';

/**
 * Gemini configuration schema
 */
export const GeminiConfigSchema = z.object({
  engine: z.literal('GEMINI').default('GEMINI'),
  modelName: z.string().min(1, 'Model name is required'),
  geminiApiKey: z.string().min(1, 'Gemini API key is required'),
  temperature: z.number().min(0).max(2).optional(),
  topP: z.number().min(0).max(1).optional(),
  topK: z.number().int().min(1).optional(),
});

export type GeminiConfig = z.infer<typeof GeminiConfigSchema>;

/**
 * Validate and parse Gemini configuration from environment
 */
export function validateGeminiConfig(): GeminiConfig {
  const config = {
    engine: 'GEMINI' as const,
    modelName: process.env.MODEL_NAME || 'gemini-2.5-flash',
    geminiApiKey: process.env.GEMINI_API_KEY || '',
    temperature: process.env.TEMPERATURE ? parseFloat(process.env.TEMPERATURE) : undefined,
    topP: process.env.TOP_P ? parseFloat(process.env.TOP_P) : undefined,
    topK: process.env.TOP_K ? parseInt(process.env.TOP_K, 10) : undefined,
  };

  return GeminiConfigSchema.parse(config);
}
