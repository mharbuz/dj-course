/**
 * ChatSession - Manages a single chat session
 */

import { v4 as uuidv4 } from 'uuid';
import type { Assistant } from '../assistant/assistant.js';
import type {
  ILLMClient,
  ILLMChatSession,
  Message,
  LLMResponse,
  TokenInfo,
  Result,
} from '../types/index.js';
import { loadSessionFileData, saveSessionHistory } from '../files/sessionFiles.js';
import { appendToWAL } from '../files/wal.js';
import { MAX_CONTEXT_TOKENS } from '../files/config.js';
import { getAssistant } from '../assistant/registry.js';
import { GeminiLLMClient } from '../llm/geminiClient.js';
import { LlamaClient } from '../llm/llamaClient.js';
import { OllamaClient } from '../llm/ollamaClient.js';
import { generateTitle, DEFAULT_TITLE } from './titleGenerator.js';

/**
 * Engine mapping for LLM client selection
 */
const ENGINE_MAPPING: Record<string, typeof GeminiLLMClient | typeof LlamaClient | typeof OllamaClient> = {
  LLAMA_CPP: LlamaClient,
  GEMINI: GeminiLLMClient,
  OLLAMA: OllamaClient,
};

/**
 * Get the selected LLM client based on ENGINE environment variable
 */
function getSelectedLLMClient(): ILLMClient {
  const engine = (process.env.ENGINE || 'GEMINI').toUpperCase();
  const SelectedClientClass = ENGINE_MAPPING[engine] || GeminiLLMClient;
  return SelectedClientClass.fromEnvironment();
}

/**
 * ChatSession class - represents and manages a single chat session
 */
export class ChatSession {
  private sessionId: string;
  private history: Message[] = [];
  private llmClient: ILLMClient;
  private llmChatSession: ILLMChatSession;
  private assistant: Assistant;
  private _title: string;
  private _titleGenerating: boolean = false;

  constructor(assistant: Assistant, sessionId?: string, history?: Message[], title?: string) {
    this.sessionId = sessionId || uuidv4();
    this.assistant = assistant;
    this.history = history || [];
    this._title = title || DEFAULT_TITLE;

    // Initialize LLM client
    this.llmClient = getSelectedLLMClient();

    // Create chat session
    this.llmChatSession = this.llmClient.createChatSession(
      assistant.systemPrompt,
      this.history
    );
  }

  /**
   * Load session from file
   */
  static loadFromFile(
    assistant: Assistant,
    sessionId: string
  ): Result<ChatSession, string> {
    const fileDataResult = loadSessionFileData(sessionId);

    if (!fileDataResult.success) {
      return { success: false, error: fileDataResult.error };
    }

    const fileData = fileDataResult.value;
    const history = fileData.history.map((msg) => ({
      role: msg.role,
      parts: [{ text: msg.text }],
    })) as Message[];

    // Restore the correct assistant from saved assistant_id (fall back to passed-in assistant)
    const savedAssistant = fileData.assistant_id
      ? getAssistant(fileData.assistant_id) ?? assistant
      : assistant;

    const session = new ChatSession(savedAssistant, sessionId, history, fileData.title);

    // Old sessions without title — schedule auto-generation from first user message
    if (!fileData.title && history.length > 0) {
      const firstUserMsg = history.find((m) => m.role === 'user');
      if (firstUserMsg) {
        session.triggerTitleGeneration(firstUserMsg.parts[0]?.text || '');
      }
    }

    return { success: true, value: session };
  }

  /**
   * Save session to file
   */
  saveToFile(): Result<boolean, string> {
    return saveSessionHistory(
      this.sessionId,
      this.history,
      this.assistant.systemPrompt,
      this.llmClient.getModelName(),
      this._title,
      this.assistant.id
    );
  }

  /**
   * Send a message and get response
   */
  async sendMessage(text: string): Promise<LLMResponse> {
    const isFirstExchange = this.history.length === 0;

    // Send message to LLM
    const response = await this.llmChatSession.sendMessage(text);

    // Sync history from LLM session (it updates internally)
    this.history = this.llmChatSession.getHistory();

    // Log to WAL
    const totalTokens = this.countTokens();
    appendToWAL(
      this.sessionId,
      text,
      response.text,
      totalTokens,
      this.llmClient.getModelName()
    );

    // Generate title after first exchange
    if (isFirstExchange && this._title === DEFAULT_TITLE) {
      this.triggerTitleGeneration(text);
    }

    return response;
  }

  /**
   * Get conversation history
   */
  getHistory(): Message[] {
    return this.history;
  }

  /**
   * Clear all history
   */
  clearHistory(): void {
    this.history = [];
    // Recreate chat session with empty history
    this.llmChatSession = this.llmClient.createChatSession(
      this.assistant.systemPrompt,
      []
    );
  }

  /**
   * Remove last user-assistant exchange
   */
  popLastExchange(): boolean {
    if (this.history.length < 2) {
      return false;
    }

    // Remove last two messages (user + assistant)
    this.history.splice(this.history.length - 2, 2);

    // Recreate chat session with updated history
    this.llmChatSession = this.llmClient.createChatSession(
      this.assistant.systemPrompt,
      this.history
    );

    return true;
  }

  /**
   * Count total tokens in history
   */
  countTokens(): number {
    return this.llmClient.countHistoryTokens(this.history);
  }

  /**
   * Check if session is empty
   */
  isEmpty(): boolean {
    return this.history.length === 0;
  }

  /**
   * Get remaining tokens in context
   */
  getRemainingTokens(): number {
    const used = this.countTokens();
    return MAX_CONTEXT_TOKENS - used;
  }

  /**
   * Get token information
   */
  getTokenInfo(): TokenInfo {
    const total = this.countTokens();
    const remaining = this.getRemainingTokens();
    return {
      total,
      remaining,
      max: MAX_CONTEXT_TOKENS,
    };
  }

  /**
   * Switch to a different assistant mid-session.
   * Inserts a marker pair in history so the model knows the switch happened,
   * then reinitializes the LLM session with the new system prompt.
   */
  switchAssistant(newAssistant: Assistant): void {
    const oldName = this.assistant.name;
    const newName = newAssistant.name;

    // Insert a marker exchange into history visible to the model
    const switchNote = `[SYSTEM: Nastąpiła zmiana asystenta z "${oldName}" na "${newName}". Od tego momentu odpowiadasz jako ${newName}.]`;
    this.history.push(
      { role: 'user', parts: [{ text: switchNote }] },
      { role: 'model', parts: [{ text: `Rozumiem. Jestem teraz ${newName}. Jak mogę pomóc?` }] }
    );

    this.assistant = newAssistant;

    // Reinitialize LLM session with the new system prompt and full history
    this.llmChatSession = this.llmClient.createChatSession(
      newAssistant.systemPrompt,
      this.history
    );
  }

  /**
   * Get the current assistant object
   */
  getAssistant(): Assistant {
    return this.assistant;
  }

  /**
   * Get assistant name
   */
  get assistantName(): string {
    return this.assistant.name;
  }

  /**
   * Get session ID
   */
  get id(): string {
    return this.sessionId;
  }

  /**
   * Get model name
   */
  get modelName(): string {
    return this.llmClient.getModelName();
  }

  /**
   * Get session title
   */
  get title(): string {
    return this._title;
  }

  /**
   * Set session title
   */
  set title(newTitle: string) {
    this._title = newTitle;
  }

  /**
   * Callback invoked when title is auto-generated
   */
  onTitleGenerated?: (title: string) => void;

  /**
   * Trigger async title generation (fire-and-forget)
   */
  triggerTitleGeneration(firstUserMessage: string): void {
    if (this._titleGenerating) return;
    this._titleGenerating = true;

    generateTitle(firstUserMessage)
      .then((title) => {
        this._title = title;
        this._titleGenerating = false;
        this.onTitleGenerated?.(title);
      })
      .catch(() => {
        this._titleGenerating = false;
      });
  }
}
