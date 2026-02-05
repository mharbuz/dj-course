/**
 * Command handler for slash commands
 */

import { displayHelp, printError, printInfo, printSuccess, printHelp } from './cli/console.js';
import { setInputHistory } from './cli/prompt.js';
import { displaySessionList } from './commands/sessionList.js';
import { displaySessionHistory } from './commands/sessionDisplay.js';
import { removeCurrentSession } from './commands/sessionRemove.js';
import { runAudioCommand } from './commands/audio.js';
import { getAssistant, listAssistants } from './assistant/registry.js';
import type { SessionManager } from './session/sessionManager.js';

/**
 * Valid slash commands
 */
const VALID_SLASH_COMMANDS = [
  '/exit',
  '/quit',
  '/switch',
  '/help',
  '/session',
  '/assistant',
  '/pdf',
  '/audio',
];

/**
 * Handle a slash command
 * @returns true if should exit, false otherwise
 */
export function handleCommand(
  userInput: string,
  manager: SessionManager
): boolean {
  const parts = userInput.trim().split(/\s+/);
  const command = parts[0].toLowerCase();
  const args = parts.slice(1);

  // Check if it's a valid command
  if (!VALID_SLASH_COMMANDS.some((cmd) => command.startsWith(cmd))) {
    printError(`Unknown command: ${command}`);
    printInfo('Type /help for available commands');
    return false;
  }

  // Handle commands
  switch (command) {
    case '/exit':
    case '/quit':
      printInfo('Do widzenia!');
      return true;

    case '/help':
      displayHelp(manager.getCurrentSession().id);
      return false;

    case '/session':
      handleSessionSubcommand(args, manager);
      return false;

    case '/switch':
      if (args.length === 0) {
        printError('Usage: /switch <SESSION_ID>');
      } else {
        const sessionId = args[0];
        const result = manager.switchToSession(sessionId);

        if (result.loadSuccessful) {
          printSuccess(`Switched to session ${sessionId}`);
          // Seed input history with user messages from loaded session
          const userMsgs = result.session
            .getHistory()
            .filter((m: any) => m.role === 'user')
            .map((m: any) => m.parts[0]?.text || '')
            .filter(Boolean);
          setInputHistory(userMsgs);
          if (result.hasHistory) {
            const session = result.session;
            const tokenInfo = session.getTokenInfo();
            printInfo(
              `Session has ${session.getHistory().length} messages (${tokenInfo.total} tokens)`
            );
          }
        } else {
          printError(`Failed to switch: ${result.error}`);
        }
      }
      return false;

    case '/assistant':
      handleAssistantSubcommand(args, manager);
      return false;

    case '/pdf':
      printError('PDF export not yet implemented');
      return false;

    case '/audio':
      runAudioCommand(manager);
      return false;

    default:
      printError(`Unknown command: ${command}`);
      return false;
  }
}

/**
 * Handle /session subcommands
 */
function handleSessionSubcommand(args: string[], manager: SessionManager): void {
  if (args.length === 0) {
    printError('Usage: /session <list|display|new|clear|pop|remove|title|rename>');
    return;
  }

  const subcommand = args[0].toLowerCase();

  switch (subcommand) {
    case 'list':
      displaySessionList();
      break;

    case 'display':
      displaySessionHistory(manager.getCurrentSession());
      break;

    case 'new':
      const createResult = manager.createNewSession(true);
      setInputHistory([]);
      printSuccess(`Created new session: ${createResult.session.id}`);
      break;

    case 'clear':
      manager.getCurrentSession().clearHistory();
      printSuccess('Session history cleared');
      break;

    case 'pop':
      if (manager.getCurrentSession().popLastExchange()) {
        printSuccess('Removed last message exchange');
      } else {
        printInfo('No messages to remove');
      }
      break;

    case 'remove':
      removeCurrentSession(manager);
      break;

    case 'title':
      printInfo(`Tytuł sesji: ${manager.getCurrentSession().title}`);
      break;

    case 'rename': {
      const newTitle = args.slice(1).join(' ').trim();
      if (!newTitle) {
        printError('Usage: /session rename <NEW_TITLE>');
        break;
      }
      manager.getCurrentSession().title = newTitle;
      const saveResult = manager.getCurrentSession().saveToFile();
      if (saveResult.success) {
        printSuccess(`Tytuł zmieniony na: ${newTitle}`);
      } else {
        printError(`Błąd zapisu: ${saveResult.error}`);
      }
      break;
    }

    default:
      printError(`Unknown subcommand: ${subcommand}`);
      printInfo('Available: list, display, new, clear, pop, remove, title, rename');
  }
}

/**
 * Handle /assistant subcommands
 */
function handleAssistantSubcommand(args: string[], manager: SessionManager): void {
  if (args.length === 0) {
    printError('Usage: /assistant <list|switch <ID>>');
    return;
  }

  const subcommand = args[0].toLowerCase();

  switch (subcommand) {
    case 'list': {
      const session = manager.getCurrentSession();
      const currentId = session.getAssistant().id;
      printHelp('\n--- Dostępni asystenci ---');
      for (const a of listAssistants()) {
        const marker = a.id === currentId ? ' (aktywny)' : '';
        printHelp(`  ${a.id} — ${a.name}${marker}`);
      }
      printHelp('-------------------------');
      break;
    }

    case 'switch': {
      if (args.length < 2) {
        printError('Usage: /assistant switch <ID>');
        printInfo('Użyj /assistant list aby zobaczyć dostępnych asystentów.');
        break;
      }
      const assistantId = args[1].toLowerCase();
      const newAssistant = getAssistant(assistantId);
      if (!newAssistant) {
        printError(`Nieznany asystent: ${assistantId}`);
        printInfo('Użyj /assistant list aby zobaczyć dostępnych asystentów.');
        break;
      }
      const session = manager.getCurrentSession();
      if (session.getAssistant().id === assistantId) {
        printInfo(`Asystent ${newAssistant.name} jest już aktywny.`);
        break;
      }
      session.switchAssistant(newAssistant);
      session.saveToFile();
      printSuccess(`Przełączono asystenta na: ${newAssistant.name}`);
      break;
    }

    default:
      printError(`Nieznana podkomenda: ${subcommand}`);
      printInfo('Dostępne: list, switch');
  }
}
