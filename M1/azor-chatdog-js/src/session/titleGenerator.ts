/**
 * Title generator — creates concise session titles using LLM
 *
 * Uses TITLE_ENGINE / TITLE_MODEL env vars to select the model.
 * Falls back to the main ENGINE / MODEL_NAME if not set.
 * On LLM failure, truncates the first user message as fallback.
 */

import type { ILLMClient } from '../types/index.js';
import { GeminiLLMClient } from '../llm/geminiClient.js';
import { OllamaClient } from '../llm/ollamaClient.js';
import { LlamaClient } from '../llm/llamaClient.js';

const DEFAULT_TITLE = 'Nowy azor-chat';
const FALLBACK_MAX_LENGTH = 40;

const TITLE_SYSTEM_PROMPT =
  'Jesteś generatorem tytułów konwersacji. Tworzysz krótkie, zwięzłe tytuły (maks. 6 słów) opisujące temat rozmowy. Odpowiadasz WYŁĄCZNIE tytułem — bez cudzysłowów, bez dodatkowego tekstu.';

const ENGINE_MAPPING: Record<string, typeof GeminiLLMClient | typeof LlamaClient | typeof OllamaClient> = {
  GEMINI: GeminiLLMClient,
  LLAMA_CPP: LlamaClient,
  OLLAMA: OllamaClient,
};

/**
 * Build a lightweight LLM client for title generation.
 * Respects TITLE_ENGINE / TITLE_MODEL env vars, falling back to main config.
 */
function getTitleLLMClient(): ILLMClient {
  const engine = (process.env.TITLE_ENGINE || process.env.ENGINE || 'GEMINI').toUpperCase();
  const SelectedClientClass = ENGINE_MAPPING[engine] || GeminiLLMClient;

  // Use the standard fromEnvironment — the model override is handled
  // by temporarily injecting TITLE_MODEL into the relevant env var.
  const titleModel = process.env.TITLE_MODEL;

  if (titleModel) {
    // Temporarily override model name for the client construction
    const envKey = engine === 'OLLAMA' ? 'OLLAMA_MODEL_NAME' : 'MODEL_NAME';
    const original = process.env[envKey];
    process.env[envKey] = titleModel;
    try {
      return SelectedClientClass.fromEnvironment();
    } finally {
      if (original !== undefined) {
        process.env[envKey] = original;
      } else {
        delete process.env[envKey];
      }
    }
  }

  return SelectedClientClass.fromEnvironment();
}

/**
 * Fallback: truncate first user message to create a title
 */
function fallbackTitle(firstUserMessage: string): string {
  const cleaned = firstUserMessage.replace(/\s+/g, ' ').trim();
  if (cleaned.length <= FALLBACK_MAX_LENGTH) {
    return cleaned;
  }
  return cleaned.substring(0, FALLBACK_MAX_LENGTH - 1) + '…';
}

/**
 * Generate a session title from the first user message using LLM.
 * Returns DEFAULT_TITLE if the message is empty.
 * Falls back to truncated message on LLM error.
 */
export async function generateTitle(firstUserMessage: string): Promise<string> {
  if (!firstUserMessage.trim()) {
    return DEFAULT_TITLE;
  }

  try {
    const client = getTitleLLMClient();
    const session = client.createChatSession(TITLE_SYSTEM_PROMPT);
    const response = await session.sendMessage(
      `Nadaj tytuł konwersacji zaczynającej się od: "${firstUserMessage}"`
    );
    // The model may output chain-of-thought before the actual title.
    // Take only the last non-empty line as the title.
    const lines = response.text.split('\n').map(l => l.trim()).filter(Boolean);
    const rawTitle = (lines[lines.length - 1] || '').replace(/["']/g, '').trim();
    return rawTitle || fallbackTitle(firstUserMessage);
  } catch {
    return fallbackTitle(firstUserMessage);
  }
}

export { DEFAULT_TITLE };
