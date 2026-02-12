/**
 * User input handling with autocompletion support
 */

import readline from 'readline';
import inquirer from 'inquirer';

/**
 * Module-level input history for readline (oldest → newest)
 */
let inputHistory: string[] = [];

/**
 * Seed input history (e.g. from loaded session's user messages)
 */
export function setInputHistory(lines: string[]): void {
  inputHistory = [...lines];
}

/**
 * Available commands and their subcommands for autocompletion
 */
const commands: Record<string, string[]> = {
  '/session': ['list', 'display', 'new', 'clear', 'pop', 'remove', 'title', 'rename'],
  '/assistant': ['list', 'switch'],
  '/switch': [],
  '/help': [],
  '/exit': [],
  '/quit': [],
  '/pdf': [],
  '/audio': [],
};

/**
 * Completer function for readline autocompletion
 */
function completer(line: string): [string[], string] {
  // Split input line into tokens
  const tokens = line.split(/\s+/);
  const lastToken = tokens[tokens.length - 1];

  // Case 1: Completing main commands (e.g., "/ses" -> "/session")
  if (tokens.length === 1 && !line.endsWith(' ')) {
    const hits = Object.keys(commands).filter((c) => c.startsWith(line));
    return [hits, line];
  }

  // Case 2: After a main command with space, suggest all subcommands
  if (tokens.length > 1 && line.endsWith(' ')) {
    const primaryCommand = tokens[0];
    const subCommands = commands[primaryCommand] || [];
    return [subCommands, lastToken];
  }

  // Case 3: Completing subcommands (e.g., "/session l" -> "list")
  if (tokens.length > 1 && !line.endsWith(' ')) {
    const primaryCommand = tokens[0];
    const subCommands = commands[primaryCommand] || [];
    const hits = subCommands.filter((s) => s.startsWith(lastToken));
    return [hits, lastToken];
  }

  // Default: no completions
  return [[], ''];
}

/**
 * Get user input with autocompletion support
 */
export async function getUserInput(prompt: string = 'TY: '): Promise<string> {
  return new Promise((resolve) => {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      completer: completer,
      terminal: true,
      history: [...inputHistory].reverse(),
    } as readline.ReadLineOptions);

    rl.question(prompt, (answer) => {
      rl.close();
      const trimmed = answer.trim();
      if (trimmed) {
        inputHistory.push(trimmed);
      }
      resolve(trimmed);
    });
  });
}

/**
 * Get clarification input from user (used by tool executor)
 * Uses inquirer instead of raw readline to avoid stdin state issues
 * after async gaps (e.g., LLM API calls between readline sessions)
 */
export async function getClarificationInput(prompt: string): Promise<string> {
  const answers = await inquirer.prompt([
    {
      type: 'input',
      name: 'answer',
      message: prompt,
    },
  ]);

  return (answers.answer as string).trim();
}

/**
 * Get confirmation from user
 */
export async function getConfirmation(message: string): Promise<boolean> {
  const answers = await inquirer.prompt([
    {
      type: 'confirm',
      name: 'confirmed',
      message,
      default: false,
    },
  ]);

  return answers.confirmed;
}

/**
 * Select from a list of options
 */
export async function selectFromList<T>(
  message: string,
  choices: Array<{ name: string; value: T }>
): Promise<T> {
  const answers = await inquirer.prompt([
    {
      type: 'list',
      name: 'selected',
      message,
      choices,
    },
  ]);

  return answers.selected;
}
