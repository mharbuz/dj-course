/**
 * Main chat loop and initialization
 */

import { config } from 'dotenv';
import { createAzorAssistant } from './assistant/azor.js';
import { getSessionManager } from './session/sessionManager.js';
import { getSessionIdFromCLI } from './cli/args.js';
import { getUserInput, setInputHistory } from './cli/prompt.js';
import { printAssistant, printInfo, printError } from './cli/console.js';
import { printWelcome } from './commands/welcome.js';
import { handleCommand } from './commandHandler.js';
import { getAllTools } from './tools/toolRegistry.js';
import { executeToolCalls } from './tools/toolExecutor.js';
import { shutdownLangfuse } from './observability/langfuse.js';

// Load environment variables
config();

/**
 * Initialize the chat application
 */
export function initChat(): void {
  // Print welcome banner
  printWelcome();

  // Create assistant
  const assistant = createAzorAssistant();

  // Get all available tools
  const tools = getAllTools();

  // Get session manager
  const manager = getSessionManager(assistant, tools);

  // Get CLI session ID if provided
  const cliSessionId = getSessionIdFromCLI();

  // Initialize session from CLI
  const session = manager.initializeFromCLI(cliSessionId);

  // Seed input history with user messages from loaded session
  const userMessages = session
    .getHistory()
    .filter((m) => m.role === 'user')
    .map((m) => m.parts[0]?.text || '')
    .filter(Boolean);
  setInputHistory(userMessages);

  // Display session info
  if (cliSessionId) {
    printInfo(`Loaded session: ${session.id} [${session.title}]`);
  } else {
    printInfo(`Started new session: ${session.id} [${session.title}]`);
  }

  printInfo(`Using model: ${session.modelName}`);
  printInfo(`Asystent: ${session.assistantName}`);
  printInfo('Type /help for available commands\n');

  // Register cleanup handler
  process.on('exit', () => {
    manager.cleanupAndSave();
  });

  process.on('SIGINT', () => {
    console.log('\n');
    printInfo('Saving session and exiting...');
    manager.cleanupAndSave();
    shutdownLangfuse().finally(() => process.exit(0));
  });
}

/**
 * Main chat loop
 */
export async function mainLoop(): Promise<void> {
  const assistant = createAzorAssistant();
  const tools = getAllTools();
  const manager = getSessionManager(assistant, tools);

  while (true) {
    try {
      // Get user input
      const userInput = await getUserInput();

      // Skip empty input
      if (!userInput) {
        continue;
      }

      // Handle commands
      if (userInput.startsWith('/')) {
        const shouldExit = await handleCommand(userInput, manager);
        if (shouldExit) {
          break;
        }
        continue;
      }

      // Get current session
      const session = manager.getCurrentSession();

      // Silent title callback — just persist, don't print
      session.onTitleGenerated = () => {
        session.saveToFile();
      };

      // Send message to LLM
      let response = await session.sendMessage(userInput);

      // Clarification loop — handle tool calls
      while (response.toolCalls && response.toolCalls.length > 0) {
        const toolResponses = await executeToolCalls(response.toolCalls);
        response = await session.sendToolResponses(toolResponses);
      }

      // Get token information
      const tokenInfo = session.getTokenInfo();

      // Display response (use session's current assistant — may have been switched)
      printAssistant(`\n${session.assistantName}: ${response.text}`);
      printInfo(
        `Tokens: ${tokenInfo.total} (Pozostało: ${tokenInfo.remaining} / ${tokenInfo.max})`
      );

      // Save session
      const saveResult = session.saveToFile();
      if (!saveResult.success && saveResult.error) {
        printError(`Error saving session: ${saveResult.error}`);
      }
    } catch (error) {
      printError(`Error: ${(error as Error).message}`);
      console.error(error);
    }
  }

  // Cleanup on exit
  manager.cleanupAndSave();
  await shutdownLangfuse();
}
