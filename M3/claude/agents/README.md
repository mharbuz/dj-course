# Subagenty w Claude Code

## Gdzie przechowywać subagenty na dysku

Claude Code ładuje subagenty z **konkretnych ścieżek**. Kolejność priorytetu (gdy kilka plików ma tę samą nazwę):

| Lokalizacja | Zakres | Priorytet | Uwagi |
|-------------|--------|-----------|--------|
| Flaga `--agents` przy uruchomieniu CLI | Tylko bieżąca sesja | 1 (najwyższy) | Nie zapisuje się na dysk, przydatne do testów |
| **`.claude/agents/`** w katalogu projektu | Tylko ten projekt | 2 | **Tu trzymaj subagenty w tym repo.** Można commitować do gita. |
| **`~/.claude/agents/`** | Wszystkie Twoje projekty | 3 | Osobiste subagenty, dostępne wszędzie |
| Katalog `agents/` w pluginie | Tam, gdzie plugin jest włączony | 4 | Subagenty dostarczane przez pluginy |

**W tym repozytorium:** pliki są w `M3/claude/agents/`. Aby Claude Code je widział w **konkretnym projekcie**, w katalogu tego projektu musi być folder `.claude/agents/` – możesz zrobić **symlink** do `M3/claude/agents` albo skopiować/symlinkować wybrane pliki.

Przykład symlinku w katalogu projektu (np. w root repo):
```bash
mkdir -p .claude
ln -sfn "$(pwd)/M3/claude/agents" .claude/agents
```

---

## Jak łączyć subagenty z Claude Code

1. **Automatyczne ładowanie**  
   Claude Code przy starcie sesji skanuje powyższe ścieżki. Jeśli w projekcie jest `.claude/agents/` z plikami `.md`, subagenty są od razu dostępne.

2. **Interfejs `/agents`**  
   W Claude Code wpisz `/agents`. Możesz:
   - przeglądać listę subagentów (wbudowane, użytkownika, projektu, z pluginów),
   - tworzyć nowe (User-level → `~/.claude/agents/`, lub w projekcie),
   - edytować/usuwać własne subagenty,
   - generować konfigurację z Claudem („Generate with Claude”).

3. **Ręczne dodanie pliku**  
   Wystarczy utworzyć plik `.md` w `.claude/agents/` (lub `~/.claude/agents/`) z poprawnym frontmatterem i treścią. Po utworzeniu pliku uruchom `/agents` albo zrestartuj sesję, żeby subagent się załadował.

---

## Co jest potrzebne do stworzenia subagenta

### 1. Plik Markdown z YAML frontmatter

**Wymagane pola:**
- **`name`** – unikalny identyfikator (małe litery, myślniki), np. `code-reviewer`
- **`description`** – kiedy Claude ma delegować do tego subagenta (na tej podstawie Claude decyduje, czy go wywołać)

**Opcjonalne pola (frontmatter):**
- **`model`** – `sonnet` | `opus` | `haiku` | `inherit` (domyślnie `inherit`)
- **`color`** – kolor tła w UI (np. `blue`, `purple`)
- **`tools`** – lista dozwolonych narzędzi (np. `Read, Grep, Glob, Bash`); brak = dziedziczy wszystkie
- **`disallowedTools`** – lista wyłączonych narzędzi
- **`permissionMode`** – `default` | `acceptEdits` | `dontAsk` | `bypassPermissions` | `plan`
- **`skills`** – lista skilli wstrzykiwanych do kontekstu subagenta
- **`hooks`** – hooki (np. PreToolUse, PostToolUse, Stop)
- **`memory`** – pamięć między sesjami: `user` | `project` | `local`

### 2. Treść pliku = system prompt

Wszystko poniżej frontmatteru to **system prompt** subagenta – instrukcje, jak ma się zachowywać. Subagent dostaje tylko ten prompt (plus podstawowe info o środowisku), nie pełny system prompt Claude Code.

### 3. Przykład minimalny

```markdown
---
name: code-reviewer
description: Ekspert od code review. Używać po zmianach w kodzie.
model: sonnet
tools: Read, Grep, Glob, Bash
---

Jesteś doświadczonym code reviewerem. Po wywołaniu:
1. Uruchom git diff i przejrzyj zmiany.
2. Skup się na jakości, bezpieczeństwie i dobrych praktykach.
3. Podaj feedback w formie: Krytyczne / Ostrzeżenia / Sugestie.
```

---

## Jak używać

- **Automatyczna delegacja:** Claude sam wybiera subagenta na podstawie `description` i Twojego polecenia.
- **Ręczne wywołanie:**  
  *„Użyj subagenta code-reviewer do przejrzenia ostatnich zmian”*  
  *„Niech subagent sql-security-optimizer przeanalizuje ten zapytanie SQL”*
- **Wyłączenie subagenta:** w ustawieniach (settings) możesz dodać do `permissions.deny` wpis np. `Task(nazwa-subagenta)`.

Więcej: [Create custom subagents – Claude Code Docs](https://code.claude.com/docs/en/sub-agents).
