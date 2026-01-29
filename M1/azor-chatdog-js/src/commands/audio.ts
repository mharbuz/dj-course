/**
 * Audio command – get latest assistant response, generate WAV via XTTS, open/play
 */

import { spawnSync } from 'child_process';
import { printError, printSuccess, printWarning } from '../cli/console.js';
import { getAudioOutputPath } from '../files/config.js';
import type { SessionManager } from '../session/sessionManager.js';
import type { Message } from '../types/index.js';
import open from 'open';

/**
 * Get the latest assistant (model) message text from history
 */
function getLastModelMessageText(history: Message[]): string | null {
  for (let i = history.length - 1; i >= 0; i--) {
    const msg = history[i];
    if (msg.role === 'model') {
      const text = msg.parts.map((p) => p.text).join('').trim();
      return text || null;
    }
  }
  return null;
}

/**
 * Run the /audio command: latest response → TTS WAV → open/play
 */
export function runAudioCommand(manager: SessionManager): void {
  const session = manager.getCurrentSession();
  const history = session.getHistory();

  const text = getLastModelMessageText(history);
  if (!text) {
    printError('No response to convert. Send a message first.');
    return;
  }

  const scriptPath = process.env.XTTS_CLI_PATH;
  if (!scriptPath?.trim()) {
    printError('TTS not configured. Set XTTS_CLI_PATH to the path to tts_cli.py (e.g. M2/text-to-speech-xtts/tts_cli.py).');
    return;
  }

  const outputPath = getAudioOutputPath(session.id);
  const pythonCmd = process.env.PYTHON_PATH || 'python3';

  const result = spawnSync(pythonCmd, [scriptPath, '--text', '-', '--output', outputPath, '--quiet'], {
    input: text,
    encoding: 'utf-8',
    maxBuffer: 10 * 1024 * 1024, // 10 MB for long responses
  });

  if (result.status !== 0) {
    const stderr = (result.stderr || result.error?.message || 'Unknown error').toString().trim();
    printError(`TTS failed: ${stderr || 'non-zero exit'}`);
    return;
  }

  printSuccess(`Audio saved: ${outputPath}`);

  open(outputPath).catch(() => {
    printWarning('Could not open the audio file with the default app. Play it manually: ' + outputPath);
  });
}
