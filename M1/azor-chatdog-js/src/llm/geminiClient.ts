/**
 * Google Gemini API LLM Client
 */

import { GoogleGenerativeAI, Content } from '@google/generative-ai';
import type {
  ILLMClient,
  ILLMChatSession,
  Message,
  LLMResponse,
} from '../types/index.js';
import { validateGeminiConfig } from './geminiValidation.js';

/**
 * Wrapper for Gemini chat session to provide universal interface
 */
class GeminiChatSessionWrapper implements ILLMChatSession {
  private geminiSession: any;
  private history: Message[] = [];

  constructor(geminiSession: any, initialHistory?: Message[]) {
    this.geminiSession = geminiSession;
    this.history = initialHistory || [];
  }

  async sendMessage(text: string): Promise<LLMResponse> {
    const result = await this.geminiSession.sendMessage(text);
    const response = result.response;
    const responseText = response.text();

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
 * Gemini LLM Client implementation
 */
/**
 * Generation parameters for Gemini
 */
interface GeminiGenerationParams {
  temperature?: number;
  topP?: number;
  topK?: number;
}

/**
 * Gemini LLM Client implementation
 */
export class GeminiLLMClient implements ILLMClient {
  private genAI: GoogleGenerativeAI;
  private modelName: string;
  private apiKey: string;
  private generationParams: GeminiGenerationParams;

  constructor(modelName: string, apiKey: string, generationParams?: GeminiGenerationParams) {
    this.modelName = modelName;
    this.apiKey = apiKey;
    this.genAI = new GoogleGenerativeAI(apiKey);
    this.generationParams = generationParams || {};
  }

  /**
   * Create client from environment variables
   */
  static fromEnvironment(): GeminiLLMClient {
    const config = validateGeminiConfig();
    return new GeminiLLMClient(config.modelName, config.geminiApiKey, {
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
    thinkingBudget?: number
  ): ILLMChatSession {
    // Convert universal Message format to Gemini Content format
    const geminiHistory: Content[] = (history || []).map((msg) => ({
      role: msg.role === 'model' ? 'model' : 'user',
      parts: msg.parts.map((part) => ({ text: part.text })),
    }));

    // Create model configuration
    const modelConfig: any = {
      model: this.modelName,
      systemInstruction: {
        role: 'system',
        parts: [{ text: systemInstruction }],
      },
    };

    // Build generation config
    const generationConfig: any = {};
    if (thinkingBudget !== undefined) {
      generationConfig.thinkingBudget = thinkingBudget;
    }
    if (this.generationParams.temperature !== undefined) {
      generationConfig.temperature = this.generationParams.temperature;
    }
    if (this.generationParams.topP !== undefined) {
      generationConfig.topP = this.generationParams.topP;
    }
    if (this.generationParams.topK !== undefined) {
      generationConfig.topK = this.generationParams.topK;
    }
    if (Object.keys(generationConfig).length > 0) {
      modelConfig.generationConfig = generationConfig;
    }

    const model = this.genAI.getGenerativeModel(modelConfig);

    const geminiSession = model.startChat({
      history: geminiHistory,
    });

    return new GeminiChatSessionWrapper(geminiSession, history);
  }

  /**
   * Count tokens in history
   */
  countHistoryTokens(history: Message[]): number {
    // Convert to Gemini format and count
    // For now, use rough estimation (will implement actual token counting)
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
    return !!this.apiKey && this.apiKey.length > 0;
  }

  preparingForUseMessage(): string {
    return `Preparing Gemini model ${this.modelName}...`;
  }

  readyForUseMessage(): string {
    const maskedKey = this.apiKey
      ? `${this.apiKey.substring(0, 8)}...${this.apiKey.substring(this.apiKey.length - 4)}`
      : 'NOT SET';
    return `Gemini ${this.modelName} ready (API Key: ${maskedKey})`;
  }
}
