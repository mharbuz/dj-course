# Integration plan: `/audio` command + M2 text-to-speech-xtts

## 1. Goal

When the user runs **`/audio`** in the chat:

1. **Get the latest assistant response** from the current session (last message with role `model`).
2. **Generate a WAV file** from that text using the XTTS solution in `M2/text-to-speech-xtts`.
3. **Save the file** in a known location and inform the user (path / success).
4. **Open/play the generated file** so the user can hear it immediately.

If there is no previous assistant reply (empty session or user never got a response), show a clear error and do not call TTS.

---

## 2. Current state

### 2.1 Chat app (M1/azor-chatdog-js)

- **Session**: `ChatSession.getHistory()` returns `Message[]`.
- **Message**: `{ role: 'user' | 'model', parts: MessagePart[] }`, `MessagePart` has `text: string`.
- **Latest response** = last message in history where `role === 'model'`. Full text = concatenation of `msg.parts[].text` (same as in `sessionDisplay.ts`).
- **Config**: `getLogDir()` → `~/.azor`, `getOutputDir()` → `~/.azor/output`. No audio-specific path yet.
- **`/audio`**: Currently calls `runAudioCommand()` with no arguments; prints "Hello World". Needs to receive `SessionManager` (or current session) to read history.

### 2.2 TTS (M2/text-to-speech-xtts)

- **Stack**: Python, Coqui TTS (XTTS v2), `torch`, `torchaudio`, `coqui-tts[codec]`, `rich`.
- **API**: `TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")` then `tts.tts_to_file(text=..., file_path=..., speaker_wav=..., language="pl")`.
- **Inputs**: text, output WAV path, speaker reference WAV (e.g. `sample-agent.wav`), language (e.g. `"pl"`).
- **Current scripts**: `run.py` (batch + animation), `run-minimalist.py` (single run, hardcoded text). **No CLI** that accepts arbitrary text and output path from outside.

---

## 3. Data flow (high level)

```
User types /audio
       ↓
commandHandler calls runAudioCommand(manager)
       ↓
Get current session → getHistory() → find last message with role === 'model'
       ↓
If none → printError("No response to convert. Send a message first.") → return
       ↓
Extract text from last model message (parts[].text)
       ↓
Choose output path (like ~/.azor/output/audio/<sessionId>-<timestamp>.wav)
       ↓
Spawn Python: tts_cli.py --text "<escaped text>" --output "<path>" [--speaker-wav ...] [--language pl]
       ↓
Python loads XTTS, generates WAV, exits 0 or non-zero
       ↓
Node: on success → printSuccess("Audio saved: <path>"); open/play the WAV with default app; on failure → printError("TTS failed: ...")
```

---

## 4. Changes in M2 (text-to-speech-xtts)

### 4.1 New script: CLI for single TTS call

Add a **CLI script** (e.g. `tts_cli.py`) that:

- **Arguments** (e.g. via `argparse`):
  - `--text` (required): text to synthesize. For long text, allow stdin or a file if needed; for v1, `--text "..."` is enough (escaping handled by Node).
  - `--output` (required): output WAV path.
  - `--speaker-wav` (optional): path to reference speaker WAV; default e.g. `sample-agent.wav` next to script.
  - `--language` (optional): default `pl`.
- **Behaviour**:
  - Load XTTS model once.
  - Call `tts_to_file(...)` with given text, output path, speaker WAV, language.
  - Exit 0 on success, non-zero on error; no Rich animation needed (optional: `--quiet` for minimal stdout so Node can parse path only).
- **Location**: `M2/text-to-speech-xtts/tts_cli.py`.

This keeps `run.py` / `run-minimalist.py` unchanged and gives the chat app a single, stable entry point.

### 4.2 Text length and escaping

- **Long text**: XTTS can handle multi-sentence; if we hit limits later, we can truncate in Node or add `--max-chars` in Python.
- **Escaping**: Node will pass text via CLI args; for complex text (quotes, newlines), prefer passing via **stdin** or a **temp file** to avoid shell escaping issues. Recommendation: `tts_cli.py` supports `--text -` to read from stdin; Node writes text to stdin of the spawned process. Alternatively `--text-file <path>` and Node writes a temp file.

---

## 5. Changes in M1 (azor-chatdog-js)

### 5.1 `runAudioCommand(manager: SessionManager)`

- **Signature**: Change to `runAudioCommand(manager: SessionManager): void` (or `Promise<void>` if we make it async).
- **Logic**:
  1. `const session = manager.getCurrentSession();`
  2. `const history = session.getHistory();`
  3. Find last message where `msg.role === 'model'`. If none, `printError('No response to convert. Send a message first.')` and return.
  4. Build `text = msg.parts.map(p => p.text).join('')` (or join with newline if desired).
  5. Resolve output path (see 5.2).
  6. Invoke TTS (see 5.3).
  7. On success: `printSuccess('Audio saved: ' + outputPath)`, then **open/play the generated WAV file** with the system default app (see 5.6). On failure: `printError('TTS failed: ' + stderr or message)`.

### 5.2 Output path for WAV

- **Option A**: Use existing `getOutputDir()` and add a subfolder, e.g. `getOutputDir() + '/audio'`, and filename e.g.  `{sessionId}-{timestamp}.wav` (keep history). Creating `~/.azor/output/audio` on first use is consistent with existing config.

Recommendation: **Option A** with a timestamped filename so each `/audio` run keeps a file: e.g. `audio/<sessionId>-<ISO timestamp>.wav`. Helper in `config.ts`: e.g. `getAudioOutputDir()` and `getAudioOutputPath(sessionId?: string)`.

### 5.3 Invoking the Python TTS script

- **Mechanism**: `child_process.spawnSync` or `spawn` (sync is simpler for a blocking TTS call; async + spinner is nicer UX but optional).
- **Paths**:
  - **Script path**: The chat app must know where `tts_cli.py` (or the M2 project) lives. Options:
    - **Env var**: e.g. `XTTS_CLI_PATH` or `TTS_SCRIPT_PATH` pointing to `.../M2/text-to-speech-xtts/tts_cli.py`. User (or setup doc) sets it.
    - **Config file**: In `~/.azor` or project, e.g. `ttsScriptPath`.
    - **Relative path**: If the app is always run from a known root (e.g. `dj-course`), we could use `path.join(__dirname, '../../M2/text-to-speech-xtts/tts_cli.py')` — fragile if repo layout or run location changes.
  - **Python**: Use `python3` or `process.env.PYTHON_PATH || 'python3'`; venv can be activated by the user before starting the chat, or we document that the TTS script must be runnable with that interpreter (e.g. `source M2/.venv/bin/activate` then run app).
- **Passing text**: Prefer **stdin** to avoid quoting: spawn with `stdin: 'pipe'`, write text, then close stdin. Python side: read from stdin when `--text -` (or similar). Fallback: temp file; Node writes, passes path to `--text-file`, deletes after.

### 5.4 commandHandler

- In the `/audio` case, call `runAudioCommand(manager)` instead of `runAudioCommand()`.

### 5.5 Optional: async and progress

- If we use `spawn` (async), we can show a "Generating audio..." message and then "Audio saved: ..." when the process exits. Sync is fine for an initial version.

### 5.6 Open/play the generated file

- After TTS succeeds, **open the generated WAV** so the user can hear it immediately.
- **Mechanism**: Use a cross-platform approach so it works on Linux, macOS, and Windows:
  - **Option A**: npm package **`open`** (e.g. `open(outputPath)`) — opens the file with the system default application (media player for WAV). Simple and portable.
  - **Option B**: OS-specific commands: `xdg-open` (Linux), `open` (macOS), `start` (Windows) via `child_process.spawn`; no extra dependency but need to branch on `process.platform`.
- **Behaviour**: Call open/play only when TTS has exited successfully; if opening fails (e.g. headless environment, no default app), log a warning but still report success and the file path so the user can play it manually.

chose option a

---

## 6. Configuration / environment

- **Chat app**:
  - **Required**: Path to TTS script (or to M2 folder so we can derive `tts_cli.py`). Recommend env var `XTTS_CLI_PATH` in `.env.example` and in docs.
  - **Optional**: `PYTHON_PATH` (e.g. `M2/.venv/bin/python`), output dir override (if we add one).
- **TTS (M2)**:
  - No new env vars strictly required; `tts_cli.py` can use default speaker WAV and language. Optional: env for default language or speaker path.

---

## 7. Edge cases and UX

| Case | Behaviour |
|------|-----------|
| No previous assistant message | Error: "No response to convert. Send a message first." |
| Empty text (model returned empty) | Treat as no response or skip TTS and warn. |
| `XTTS_CLI_PATH` not set | Error: "TTS not configured. Set XTTS_CLI_PATH to the path to tts_cli.py." |
| Python or script not found | Error: "TTS script not found or Python not available." |
| TTS process fails (e.g. OOM, file write error) | Show stderr or a short message: "TTS failed: ...". |
| Very long text | Use as-is for v1; optionally truncate with a warning (e.g. first 5000 chars). |
| Open/play fails (headless, no default app) | Log warning; still report success and path so user can play manually. |

---

## 8. Summary checklist

**M2 (text-to-speech-xtts):**

- [ ] Add `tts_cli.py` with `--text` (or stdin), `--output`, optional `--speaker-wav`, `--language`.
- [ ] Use existing XTTS API; exit 0 on success, non-zero on failure; minimal stdout when used by Node (or document format).

**M1 (azor-chatdog-js):**

- [ ] Add `getAudioOutputDir()` (and optionally `getAudioOutputPath(sessionId)`) in `config.ts`.
- [ ] Change `runAudioCommand()` to `runAudioCommand(manager: SessionManager)`; get last model message; extract text; if none, error and return.
- [ ] Resolve TTS script path from env (e.g. `XTTS_CLI_PATH`); spawn Python with text via stdin (or temp file) and output path; handle success/failure and print result.
- [ ] After TTS success, open/play the generated WAV with default app (e.g. npm `open` or OS command); on open failure, warn but still report path.
- [ ] Update `commandHandler` to pass `manager` into `runAudioCommand(manager)`.
- [ ] Document in README or `.env.example`: `XTTS_CLI_PATH`, optional `PYTHON_PATH`, and that M2 venv/deps must be installed.

---

## 9. Review and risks

- **Dependency on Python/Coqui**: Users need Python, venv, and Coqui TTS installed for M2. First run of XTTS loads the model (slow); subsequent calls are faster. Document clearly.
- **Cross-stack**: Node spawning Python is simple and keeps TTS in Python where the library lives; no need to reimplement XTTS in Node.
- **Path and portability**: Using an env var for the script path keeps the chat app agnostic of repo layout and OS; relative paths from `__dirname` are possible but brittle.
- **Text encoding**: Use UTF-8 for stdin/file and for Node → Python; avoid issues with Polish characters.
- **Sync vs async**: Starting with sync keeps the implementation small; we can later switch to async and add a "Generating…" message if needed.
- **Playback**: Opening the WAV with the default app gives immediate feedback; in headless/CI environments opening may fail — treat as non-fatal and still show the file path.

This plan is enough to implement the integration end-to-end and then iterate on UX (e.g. async, truncation).
