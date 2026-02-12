/**
 * Tool registry — definitions of all tools available to the LLM
 */

import type { ToolDefinition } from '../types/index.js';

export const ASK_CLARIFICATION_TOOL: ToolDefinition = {
  name: 'ask_user_clarification',
  description:
    'Użyj gdy pytanie użytkownika jest niejasne, zbyt ogólne lub wieloznaczne. Zamiast zgadywać, poproś o doprecyzowanie.',
  parameters: {
    question: {
      type: 'string',
      description: 'Konkretne pytanie wyjaśniające do użytkownika',
      required: true,
    },
  },
};

export function getAllTools(): ToolDefinition[] {
  return [ASK_CLARIFICATION_TOOL];
}
