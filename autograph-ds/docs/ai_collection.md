# AI Sample Collection

This document describes how the AI sample collector (`generate_ai_samples.py`) works, the sidecar metadata schema it writes, and the environment variables that control providers and execution.

## Sidecar Schema

For each collected code sample (e.g., `ai_gpt4o_3.py`), the collector writes a JSON sidecar next to it: `ai_gpt4o_3.py.json`.

Fields:
- `identity`: AI identity key used in the dataset (e.g., `gpt4o`, `claude`, `gemini`, `deepseek_v3`, `kimi`).
- `model`: Provider model name used for generation (e.g., `gpt-4o`, `claude-sonnet-4-6`).
- `prompt_index`: Integer index of the prompt that produced this sample. Index is deterministic and stable across runs.
- `prompt`: The exact text of the prompt sent to the provider.
- `collected_at`: Unix timestamp (seconds) when the sample was collected.
- `content_hash`: First 16 hex characters of the SHA-256 hash of the code contents; used for cross-provider deduplication.
- `total_lines`: Total number of lines in the saved file (after basic cleanup).
- `code_lines`: Count of non-empty, non-comment lines as a basic quality metric.

Notes:
- The index `N` in `ai_<identity>_N.py` is fixed for a given prompt, ensuring deterministic growth when new prompts are appended to the prompt list.
- Hashing is performed on the normalized code (markdown fences stripped, leading/trailing blank lines removed).

## Execution Controls (Environment Variables)

Set these in your project-level `.env` (or your shell) before running the collector.

Required API keys (use the ones you need):
- `OPENAI_API_KEY` – Enables GPT‑4o samples via the OpenAI SDK.
- `ANTHROPIC_API_KEY` – Enables Claude Sonnet samples via the Anthropic SDK.
- `GEMINI_API_KEY` – Enables Gemini samples via the `google.genai` SDK.
- `DEEPSEEK_API_KEY` – Enables DeepSeek V3 samples via HTTPS REST.
- `OPENCODE_API_KEY` – Enables Kimi K2 samples via the OpenCode Zen gateway.

Optional feature flags:
- `AI_DRY_RUN` – If `1`/`true`/`yes`/`on`, the collector performs a dry-run: it calls providers and validates samples but does not write any files, and does not mutate existing data. The console will show which files would have been written.
- `AI_TARGET_PER_MODEL` – Integer override for how many samples to collect per provider (default: 15).
- `AI_PROVIDERS` – Comma-separated allowlist to restrict providers. Supported tokens (case-insensitive): `openai`, `gpt4o`, `anthropic`, `claude`, `gemini`, `deepseek`, `deepseek_v3`, `kimi`, `opencode`.
  - Example: `AI_PROVIDERS=openai,anthropic` limits collection to GPT‑4o and Claude.

## Rate-limit Logging

Rate-limit conditions (e.g., HTTP 429 or provider-specific errors) are detected and logged distinctly in the console as `RATE-LIMIT` entries. These are also surfaced in the per-provider summary under `skipped.rate_limited`.

## Summaries

At the end of a run, the collector prints a summary by provider and an overall roll-up including:
- Collected samples this run (or planned in dry-run mode)
- Skips by reason: `too_short`, `invalid_python`, `duplicate`, `error`, `rate_limited`
- On-disk counts per provider vs. target

This summary is intended for quick operator feedback and auditability of each collection run.

