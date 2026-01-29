/**
 * Ollama API LLM Client
 */

import type {
  ILLMClient,
  ILLMChatSession,
  Message,
  LLMResponse,
} from '../types/index.js';
import { validateOllamaConfig } from './ollamaValidation.js';

/**
 * Ollama message format
 */
interface OllamaMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

/**
 * Ollama generation options
 */
interface OllamaOptions {
  temperature?: number;
  top_p?: number;
  top_k?: number;
}

/**
 * Generation parameters (internal format)
 */
interface OllamaGenerationParams {
  temperature?: number;
  topP?: number;
  topK?: number;
}

/**
 * Ollama chat request format
 */
interface OllamaChatRequest {
  model: string;
  messages: OllamaMessage[];
  stream: boolean;
  options?: OllamaOptions;
}

/**
 * Ollama chat response format
 */
interface OllamaChatResponse {
  message: OllamaMessage;
  eval_count?: number;
}

/**
 * Convert universal Message format to Ollama format
 */
function convertToOllamaMessage(msg: Message): OllamaMessage {
  return {
    role: msg.role === 'model' ? 'assistant' : 'user',
    content: msg.parts.map((part) => part.text).join(''),
  };
}

/**
 * Wrapper for Ollama chat session to provide universal interface
 */
class OllamaChatSessionWrapper implements ILLMChatSession {
  private baseUrl: string;
  private modelName: string;
  private systemInstruction: string;
  private history: Message[] = [];
  private generationParams: OllamaGenerationParams;

  constructor(
    baseUrl: string,
    modelName: string,
    systemInstruction: string,
    initialHistory?: Message[],
    generationParams?: OllamaGenerationParams
  ) {
    this.baseUrl = baseUrl;
    this.modelName = modelName;
    this.systemInstruction = systemInstruction;
    this.history = initialHistory || [];
    this.generationParams = generationParams || {};
  }

  async sendMessage(text: string): Promise<LLMResponse> {
    // Build messages array with system instruction, history, and new message
    const messages: OllamaMessage[] = [
      { role: 'system', content: this.systemInstruction },
      ...this.history.map(convertToOllamaMessage),
      { role: 'user', content: text },
    ];

    // Build options object for generation parameters
    const options: OllamaOptions = {};
    if (this.generationParams.temperature !== undefined) {
      options.temperature = this.generationParams.temperature;
    }
    if (this.generationParams.topP !== undefined) {
      options.top_p = this.generationParams.topP;
    }
    if (this.generationParams.topK !== undefined) {
      options.top_k = this.generationParams.topK;
    }

    const request: OllamaChatRequest = {
      model: this.modelName,
      messages,
      stream: false,
      ...(Object.keys(options).length > 0 && { options }),
    };

    const response = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
    }

    const data = (await response.json()) as OllamaChatResponse;
    const responseText = data.message.content;

    // Add to history
    this.history.push({
      role: 'user',
      parts: [{ text }],
    });
    this.history.push({
      role: 'model',
      parts: [{ text: responseText }],
    });

    return { text: responseText };
  }

  getHistory(): Message[] {
    return this.history;
  }
}

/**
 * Ollama LLM Client implementation
 */
export class OllamaClient implements ILLMClient {
  private baseUrl: string;
  private modelName: string;
  private generationParams: OllamaGenerationParams;

  constructor(baseUrl: string, modelName: string, generationParams?: OllamaGenerationParams) {
    this.baseUrl = baseUrl;
    this.modelName = modelName;
    this.generationParams = generationParams || {};
  }

  /**
   * Create client from environment variables
   */
  static fromEnvironment(): OllamaClient {
    const config = validateOllamaConfig();
    return new OllamaClient(config.baseUrl, config.modelName, {
      temperature: config.temperature,
      topP: config.topP,
      topK: config.topK,
    });
  }

  /**
   * Create a chat session
   */
  createChatSession(
    systemInstruction: string,
    history?: Message[],
    _thinkingBudget?: number
  ): ILLMChatSession {
    return new OllamaChatSessionWrapper(
      this.baseUrl,
      this.modelName,
      systemInstruction,
      history,
      this.generationParams
    );
  }

  /**
   * Count tokens in history
   */
  countHistoryTokens(history: Message[]): number {
    let totalTokens = 0;
    for (const msg of history) {
      for (const part of msg.parts) {
        // Rough estimation: 1 token ≈ 4 characters
        totalTokens += Math.ceil(part.text.length / 4);
      }
    }
    return totalTokens;
  }

  getModelName(): string {
    return this.modelName;
  }

  isAvailable(): boolean {
    return !!this.baseUrl && this.baseUrl.length > 0;
  }

  preparingForUseMessage(): string {
    return `Preparing Ollama model ${this.modelName}...`;
  }

  readyForUseMessage(): string {
    return `Ollama ${this.modelName} ready (URL: ${this.baseUrl})`;
  }
}
