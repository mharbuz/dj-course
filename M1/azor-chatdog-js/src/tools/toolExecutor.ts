/**
 * Tool executor — handles execution of tool calls from LLM
 */

import type { ToolCall, ToolResponse } from '../types/index.js';
import { printClarification } from '../cli/console.js';
import { getClarificationInput } from '../cli/prompt.js';

/**
 * Execute tool calls and return responses
 */
export async function executeToolCalls(toolCalls: ToolCall[]): Promise<ToolResponse[]> {
  const responses: ToolResponse[] = [];

  for (const toolCall of toolCalls) {
    switch (toolCall.name) {
      case 'ask_user_clarification': {
        const question = toolCall.args.question as string;
        printClarification(`\n${question}`);
        const answer = await getClarificationInput('Twoja odpowiedź');
        responses.push({
          name: toolCall.name,
          result: { answer },
        });
        break;
      }
      default:
        throw new Error(`Unknown tool: ${toolCall.name}`);
    }
  }

  return responses;
}
