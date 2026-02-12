import fs from 'fs';
import path from 'path';
import os from 'os';

// --- Types ---

export interface TimestampedMessage {
  role: 'user' | 'model';
  timestamp: string;
  text: string;
}

export interface SessionHistoryFile {
  session_id: string;
  model: string;
  system_role: string;
  title?: string;
  assistant_id?: string;
  history: TimestampedMessage[];
}

export interface SessionMetadata {
  session_id: string;
  model: string;
  message_count: number;
  last_modified: Date;
  first_message?: string;
  title?: string;
  assistant_id?: string;
}

export type Result<T, E = string> =
  | { success: true; value: T }
  | { success: false; error: E };

// --- Functions ---

export function getLogDir(): string {
  const homeDir = os.homedir();
  const logDir = path.join(homeDir, '.azor');

  if (!fs.existsSync(logDir)) {
    fs.mkdirSync(logDir, { recursive: true });
  }

  return logDir;
}

function getSessionFilePath(sessionId: string): string {
  return path.join(getLogDir(), `${sessionId}-log.json`);
}

export function loadSessionFileData(
  sessionId: string
): Result<SessionHistoryFile> {
  const filePath = getSessionFilePath(sessionId);

  if (!fs.existsSync(filePath)) {
    return { success: false, error: `Session file not found: ${sessionId}` };
  }

  try {
    const fileContent = fs.readFileSync(filePath, 'utf-8');
    const data: SessionHistoryFile = JSON.parse(fileContent);
    return { success: true, value: data };
  } catch (error) {
    if (error instanceof SyntaxError) {
      return { success: false, error: `Invalid JSON in session file: ${sessionId}` };
    }
    return { success: false, error: `Error loading session: ${(error as Error).message}` };
  }
}

export function listSessions(): SessionMetadata[] {
  const logDir = getLogDir();
  const sessions: SessionMetadata[] = [];

  try {
    const files = fs.readdirSync(logDir);

    for (const file of files) {
      if (file.endsWith('-log.json')) {
        const filePath = path.join(logDir, file);
        const stats = fs.statSync(filePath);

        try {
          const content = fs.readFileSync(filePath, 'utf-8');
          const data: SessionHistoryFile = JSON.parse(content);

          sessions.push({
            session_id: data.session_id,
            model: data.model,
            message_count: data.history.length,
            last_modified: stats.mtime,
            first_message: data.history[0]?.text,
            title: data.title,
            assistant_id: data.assistant_id,
          });
        } catch {
          continue;
        }
      }
    }

    sessions.sort((a, b) => b.last_modified.getTime() - a.last_modified.getTime());
    return sessions;
  } catch {
    return [];
  }
}

export function removeSessionFile(sessionId: string): Result<boolean> {
  const filePath = getSessionFilePath(sessionId);

  if (!fs.existsSync(filePath)) {
    return { success: false, error: `Session file not found: ${sessionId}` };
  }

  try {
    fs.unlinkSync(filePath);
    return { success: true, value: true };
  } catch (error) {
    return { success: false, error: `Error removing session: ${(error as Error).message}` };
  }
}
