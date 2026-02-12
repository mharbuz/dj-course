import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import { listSessions, loadSessionFileData, removeSessionFile } from './sessions.js';

const server = new McpServer({
  name: 'azor-sessions',
  version: '1.0.0',
});

// Tool 1: list_sessions
server.tool(
  'list_sessions',
  'Listuje wszystkie zapisane sesje Azora z ~/.azor/',
  {},
  async () => {
    const sessions = listSessions();

    if (sessions.length === 0) {
      return { content: [{ type: 'text', text: 'Brak zapisanych sesji.' }] };
    }

    const header = 'ID sesji | Tytuł | Ostatnia modyfikacja | Wiadomości | Model | Asystent';
    const separator = '--- | --- | --- | --- | --- | ---';

    const rows = sessions.map((s) => {
      const id = s.session_id;
      const title = s.title || '(brak)';
      const date = s.last_modified.toISOString().slice(0, 19).replace('T', ' ');
      const count = String(s.message_count);
      const model = s.model;
      const assistant = s.assistant_id || '(brak)';
      return `${id} | ${title} | ${date} | ${count} | ${model} | ${assistant}`;
    });

    const table = [header, separator, ...rows].join('\n');
    const summary = `Znaleziono ${sessions.length} sesji.\n\n${table}`;

    return { content: [{ type: 'text', text: summary }] };
  }
);

// Tool 2: get_session
server.tool(
  'get_session',
  'Wyświetla metadane sesji oraz pierwszą i ostatnią wiadomość',
  { session_id: z.string().describe('ID sesji (UUID)') },
  async ({ session_id }) => {
    const result = loadSessionFileData(session_id);

    if (!result.success) {
      return { content: [{ type: 'text', text: `Błąd: ${result.error}` }] };
    }

    const data = result.value;
    const history = data.history;

    const lines: string[] = [
      `## Sesja: ${data.session_id}`,
      '',
      `- **Tytuł**: ${data.title || '(brak)'}`,
      `- **Model**: ${data.model}`,
      `- **Asystent**: ${data.assistant_id || '(brak)'}`,
      `- **Liczba wiadomości**: ${history.length}`,
    ];

    if (history.length > 0) {
      const first = history[0];
      const firstText = first.text.length > 500 ? first.text.slice(0, 500) + '...' : first.text;
      lines.push(
        '',
        '### Pierwsza wiadomość',
        `- **Rola**: ${first.role}`,
        `- **Czas**: ${first.timestamp}`,
        `- **Tekst**: ${firstText}`,
      );
    }

    if (history.length > 1) {
      const last = history[history.length - 1];
      const lastText = last.text.length > 500 ? last.text.slice(0, 500) + '...' : last.text;
      lines.push(
        '',
        '### Ostatnia wiadomość',
        `- **Rola**: ${last.role}`,
        `- **Czas**: ${last.timestamp}`,
        `- **Tekst**: ${lastText}`,
      );
    }

    return { content: [{ type: 'text', text: lines.join('\n') }] };
  }
);

// Tool 3: delete_session
server.tool(
  'delete_session',
  'Usuwa sesję Azora po ID',
  { session_id: z.string().describe('ID sesji do usunięcia (UUID)') },
  async ({ session_id }) => {
    const result = removeSessionFile(session_id);

    if (!result.success) {
      return { content: [{ type: 'text', text: `Błąd: ${result.error}` }] };
    }

    return { content: [{ type: 'text', text: `Sesja ${session_id} została usunięta.` }] };
  }
);

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
