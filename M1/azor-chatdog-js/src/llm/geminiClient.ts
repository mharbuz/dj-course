/**
 * Google Gemini API LLM Client
 */

import { GoogleGenerativeAI, Content } from '@google/generative-ai';
import type {
  ILLMClient,
  ILLMChatSession,
  Message,
  LLMResponse,
  ToolDefinition,
  ToolResponse,
} from '../types/index.js';
import { validateGeminiConfig } from './geminiValidation.js';

/**
 * Convert our ToolDefinition[] to Gemini functionDeclarations format
 */
function convertToolsToGeminiFormat(tools: ToolDefinition[]): any[] {
  return tools.map((tool) => {
    const properties: Record<string, any> = {};
    const required: string[] = [];

    for (const [paramName, param] of Object.entries(tool.parameters)) {
      properties[paramName] = {
        type: param.type.toUpperCase(),
        description: param.description,
      };
      if (param.required) {
        required.push(paramName);
      }
    }

    return {
      name: tool.name,
      description: tool.description,
      parameters: {
        type: 'OBJECT',
        properties,
        required,
      },
    };
  });
}

/**
 * Wrapper for Gemini chat session to provide universal interface
 */
class GeminiChatSessionWrapper implements ILLMChatSession {
  private geminiSession: any;
  private history: Message[] = [];
  private pendingUserText?: string;

  constructor(geminiSession: any, initialHistory?: Message[]) {
    this.geminiSession = geminiSession;
    this.history = initialHistory || [];
  }

  async sendMessage(text: string): Promise<LLMResponse> {
    const result = await this.geminiSession.sendMessage(text);
    const response = result.response;

    // Check for function calls
    const functionCalls = response.functionCalls?.();
    if (functionCalls && functionCalls.length > 0) {
      // Save pending user text — do NOT add to history yet
      this.pendingUserText = text;
      return {
        text: '',
        toolCalls: functionCalls.map((fc: any) => ({
          name: fc.name,
          args: fc.args || {},
        })),
      };
    }

    // Normal text response
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

  async sendToolResponses(toolResponses: ToolResponse[]): Promise<LLMResponse> {
    // Build function response parts for Gemini
    const functionResponseParts = toolResponses.map((tr) => ({
      functionResponse: {
        name: tr.name,
        response: tr.result,
      },
    }));

    const result = await this.geminiSession.sendMessage(functionResponseParts);
    const response = result.response;
    const responseText = response.text();

    // Now add to history: user (pending) + model (final)
    if (this.pendingUserText !== undefined) {
      this.history.push({
        role: 'user',
        parts: [{ text: this.pendingUserText }],
      });
    }
    this.history.push({
      role: 'model',
      parts: [{ text: responseText }],
    });

    this.pendingUserText = undefined;

    return { text: responseText };
  }

  getHistory(): Message[] {
    return this.history;
  }
}

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
    thinkingBudget?: number,
    tools?: ToolDefinition[]
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

    // Add tools as functionDeclarations
    if (tools && tools.length > 0) {
      modelConfig.tools = [
        { functionDeclarations: convertToolsToGeminiFormat(tools) },
      ];
    }

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
