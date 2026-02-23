"""
Generate AI Code Samples from Real LLM APIs

This script fetches genuine code samples from various AI models using their APIs.
It replaces synthetic stubs with real model outputs for authentic attestation training.

Robustness features:
  - Idempotent: re-running will only collect missing samples (never overwrites)
  - Incremental: each file is saved immediately after collection
  - Deterministic: prompt N always maps to sample index N (no random.sample)
  - Deduplicated: content-hashed to prevent near-duplicate samples across runs
  - Validated: syntax-checked and minimum line count enforced
  - Auditable: sidecar .json written per sample (model, prompt, hash, timestamp)

Required environment variables (in .env file):
    OPENAI_API_KEY    - For GPT-4o samples
    ANTHROPIC_API_KEY - For Claude samples
    GEMINI_API_KEY    - For Gemini samples
    DEEPSEEK_API_KEY  - For DeepSeek V3 samples (platform.deepseek.com)
    OPENCODE_API_KEY  - For Kimi K2 samples via OpenCode Zen (opencode.ai)
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

import requests

# Load .env from project root (parent of autograph-ds)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Optional LLM clients
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


# --- Configuration ---

TARGET_PER_MODEL = 55  # Increased from 15 to match human author sample distribution (~50-100)
MIN_LINES = 3        # total lines (splitlines)
MIN_CODE_LINES = 3   # non-blank, non-comment lines

# Fixed ordered prompts — index N always produces sample N.
# Add new prompts at the END to avoid shifting existing assignments.
PROMPTS = [
    "Write a Python function to calculate the Fibonacci sequence up to N.",
    "Write a Python class for a simple Bank Account with deposit and withdraw methods.",
    "Write a Python function that parses a CSV string and returns a list of dictionaries.",
    "Write a Python script to find all files with a .py extension in a directory recursively.",
    "Write a Python function to check if a string is a palindrome.",
    "Write a Python function that implements a binary search algorithm.",
    "Write a Python decorator that measures the execution time of a function.",
    "Write a Python function to merge two sorted lists.",
    "Write a Python script that fetches data from a JSON API using requests.",
    "Write a Python function to find the most frequent element in a list.",
    "Write a Python class representing a priority queue using heapq.",
    "Write a Python function to compute the Levenshtein distance between two strings.",
    "Write a Python context manager for temporary file handling.",
    "Write a Python generator function that yields prime numbers up to N.",
    "Write a Python dataclass representing a 3D point with distance calculation.",
    "Write a Python function to flatten a nested list of arbitrary depth.",
    "Write a Python async function that fetches multiple URLs concurrently.",
    "Write a Python function to validate an email address using regex.",
    "Write a Python class implementing a LRU cache.",
    "Write a Python function to parse and evaluate simple mathematical expressions.",
    "Write a Python script to monitor a directory for file changes.",
    "Write a Python function to compress a string using run-length encoding.",
    "Write a Python class representing a thread-safe counter.",
    "Write a Python function to find all anagrams of a word in a dictionary.",
    "Write a Python decorator that implements retry logic with exponential backoff.",
    # --- Web Development & APIs ---
    "Write a Flask route with JWT authentication that validates a user's token and returns protected data.",
    "Write a FastAPI endpoint that accepts file uploads and saves them with validation.",
    "Write a Python function that makes async HTTP requests to multiple APIs concurrently and aggregates results.",
    "Write a Python decorator that implements rate limiting for API endpoints.",
    "Write a Python function that implements OAuth2 token refresh logic with automatic retry.",
    "Write a Python class for a simple REST API client with automatic retry and error handling.",
    "Write a Python function that parses and validates webhook payloads with signature verification.",
    "Write a Python script that scrapes a webpage using BeautifulSoup and extracts structured data.",
    "Write a Python function that implements pagination logic for API responses.",
    "Write a Python class that manages WebSocket connections with automatic reconnection.",
    # --- Data Processing & Analysis ---
    "Write a Python function that reads a large CSV file in chunks and performs aggregation.",
    "Write a Python function that transforms nested JSON data into a flat dictionary structure.",
    "Write a Python class that implements a simple ETL pipeline with data validation.",
    "Write a Python function that performs time-series data resampling and interpolation.",
    "Write a Python function that detects outliers in a dataset using statistical methods.",
    "Write a Python script that generates a PDF report from data using a templating engine.",
    "Write a Python function that validates and sanitizes user input for SQL queries (prevent injection).",
    "Write a Python class that implements a circular buffer for streaming data.",
    "Write a Python function that performs fuzzy string matching across large datasets.",
    "Write a Python function that implements a bloom filter for efficient membership testing.",
    # --- System & Automation ---
    "Write a Python script that monitors a log file in real-time and alerts on patterns.",
    "Write a Python function that implements a simple task scheduler with priority queue.",
    "Write a Python class for a thread-safe message queue with blocking operations.",
    "Write a Python function that parallelizes CPU-intensive work using multiprocessing.",
    "Write a Python script that syncs files between directories with conflict resolution.",
    "Write a Python function that implements circuit breaker pattern for external service calls.",
    "Write a Python context manager that temporarily modifies environment variables.",
    "Write a Python function that generates unique IDs with distributed system support.",
    "Write a Python class that implements a simple key-value store with persistence.",
    "Write a Python function that handles graceful shutdown with signal handling.",
    # --- Testing & Quality ---
    "Write a Python function that generates mock data using the Faker library.",
    "Write a Python decorator that logs function entry, exit, and execution time.",
    "Write a Python class that implements a simple test runner with assertion tracking.",
    "Write a Python function that performs deep equality comparison of nested structures.",
    "Write a Python fixture for pytest that sets up and tears down a database connection.",
    "Write a Python function that generates property-based test cases using Hypothesis.",
    "Write a Python context manager that captures and suppresses specific exceptions for testing.",
    "Write a Python function that measures code coverage for a given module.",
    "Write a Python class that implements a mock HTTP server for testing API clients.",
    "Write a Python function that validates JSON schemas against provided data.",
    # --- Advanced Algorithms & Data Structures ---
    "Write a Python class that implements a trie data structure with prefix search.",
    "Write a Python function that solves the knapsack problem using dynamic programming.",
    "Write a Python class that implements a graph with BFS and DFS traversal methods.",
    "Write a Python function that finds the shortest path in a weighted graph using Dijkstra's algorithm.",
    "Write a Python class that implements an LRU cache with O(1) operations.",
    "Write a Python function that performs topological sorting of a directed acyclic graph.",
    "Write a Python class that implements a skip list for efficient searching.",
    "Write a Python function that implements the A* pathfinding algorithm.",
    "Write a Python function that solves the traveling salesman problem using simulated annealing.",
    "Write a Python class that implements a segment tree for range queries.",
    # --- Error Handling & Edge Cases ---
    "Write a Python function that handles timeout scenarios with multiple fallback strategies.",
    "Write a Python class that implements exponential backoff retry with jitter.",
    "Write a Python function that validates email addresses with comprehensive regex and DNS check.",
    "Write a Python context manager that handles resource cleanup on exceptions.",
    "Write a Python function that implements deadlock detection and resolution for locks.",
    "Write a Python class that implements a saga pattern for distributed transactions.",
    "Write a Python function that sanitizes HTML to prevent XSS attacks.",
    "Write a Python function that implements idempotency keys for API operations.",
    "Write a Python class that manages database transactions with rollback on failure.",
    "Write a Python function that implements optimistic locking for concurrent updates.",
    # --- Configuration & CLI ---
    "Write a Python function that parses command-line arguments using argparse with subcommands.",
    "Write a Python class that loads and validates configuration from YAML files with environment overrides.",
    "Write a Python script that implements a simple CLI progress bar with ETA calculation.",
    "Write a Python function that implements feature flags with gradual rollout logic.",
    "Write a Python class that manages application secrets with environment-based encryption.",
    # --- Database & ORM ---
    "Write a Python class that implements an in-memory SQLite database manager with connection pooling.",
    "Write a Python function that performs database migrations with schema versioning.",
    "Write a Python class that implements a simple ORM with CRUD operations.",
    "Write a Python function that implements connection pooling for database connections.",
    "Write a Python script that performs database backup and restoration.",
    # --- Caching & Performance ---
    "Write a Python class that implements a TTL cache with automatic expiration.",
    "Write a Python function that implements memoization with cache size limits.",
    "Write a Python class that implements a simple load balancer with round-robin strategy.",
    "Write a Python function that performs batch processing with configurable chunk sizes.",
    "Write a Python class that implements a throttling mechanism for rate-limited operations.",
    # --- Security & Cryptography ---
    "Write a Python function that implements password hashing with salt using bcrypt.",
    "Write a Python function that generates cryptographically secure random tokens.",
    "Write a Python class that implements a simple HMAC signature verifier.",
    "Write a Python function that performs file encryption and decryption using Fernet.",
    "Write a Python class that implements role-based access control (RBAC).",
    # --- Serialization & Protocols ---
    "Write a Python function that serializes Python objects to JSON with datetime handling.",
    "Write a Python class that implements a simple message protocol with length-prefix framing.",
    "Write a Python function that parses and generates CSV files with custom delimiters.",
    "Write a Python class that implements a simple binary protocol parser.",
    "Write a Python function that performs base64 encoding and decoding with URL-safe alphabet.",
]


# --- Utility Functions ---

class RateLimitError(Exception):
    """Raised when a provider signals a rate limit condition (HTTP 429, etc)."""


class TruncationError(Exception):
    """Raised when a provider returns a truncated response (stop_reason=max_tokens/length)."""


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _python_syntax_error(code: str) -> "Optional[SyntaxError]":
    """Returns None if the code is valid Python, or the SyntaxError if not."""
    try:
        compile(code, "<string>", "exec")
        return None
    except SyntaxError as e:
        return e


def _count_code_lines(code: str) -> int:
    count = 0
    for line in code.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            count += 1
    return count


def clean_code(code: str) -> str:
    """Strip markdown fences and leading/trailing blank lines."""
    code = code.replace("```python", "").replace("```", "").strip()
    lines = code.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


# --- Per-Provider Single-Call Fetch Functions ---
# Each returns a callable fetch_fn(prompt: str) -> str (raw LLM output).
# Returns None if the provider is unavailable (missing key or SDK).

def _make_openai_fetcher(api_key, model):
    if not api_key or not OpenAI:
        return None
    client = OpenAI(api_key=api_key)

    def fetch(prompt):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"{prompt} Output ONLY the Python code, no explanation."}],
                temperature=0.7,
            )
            return r.choices[0].message.content
        except Exception as e:  # Defer precise typing to avoid hard dep
            # Detect OpenAI rate limit patterns without importing error types
            if "rate limit" in str(e).lower() or "429" in str(e):
                raise RateLimitError(str(e))
            raise

    return fetch


def _make_anthropic_fetcher(api_key, model):
    if not api_key or not anthropic:
        return None
    client = anthropic.Anthropic(api_key=api_key)

    def fetch(prompt):
        try:
            r = client.messages.create(
                model=model,
                max_tokens=8192,
                messages=[{"role": "user", "content": f"{prompt} Output ONLY the Python code, no explanation."}],
            )
            if r.stop_reason == "max_tokens":
                raise TruncationError(
                    f"Response truncated at {len(r.content[0].text)} chars (stop_reason=max_tokens)"
                )
            return r.content[0].text
        except (RateLimitError, TruncationError):
            raise
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                raise RateLimitError(str(e))
            raise

    return fetch


def _make_gemini_fetcher(api_key, model):
    if not api_key or not genai:
        return None
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=(
            "You are an expert Python developer. "
            "Provide only high-quality, PEP 8 compliant code without explanations."
        ),
        temperature=0.3,
    )

    def fetch(prompt):
        try:
            r = client.models.generate_content(
                model=model,
                contents=f"{prompt} Output ONLY the Python code, no explanation.",
                config=config,
            )
            return r.text
        except Exception as e:
            # google.genai exceptions are not well-documented yet; pattern match
            if ("rate" in str(e).lower() and "limit" in str(e).lower()) or "429" in str(e):
                raise RateLimitError(str(e))
            raise

    return fetch


def _make_deepseek_fetcher(api_key, model):
    if not api_key:
        return None

    def fetch(prompt):
        r = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": f"{prompt} Output ONLY the Python code, no explanation."}],
                "temperature": 0.7,
            },
            timeout=60,
        )
        if r.status_code == 429:
            raise RateLimitError(f"HTTP 429: {r.text}")
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    return fetch


def _make_kimi_fetcher(api_key, model):
    """Kimi K2 via OpenCode Zen (OpenAI-compatible gateway)."""
    if not api_key or not OpenAI:
        return None
    client = OpenAI(api_key=api_key, base_url="https://opencode.ai/zen/v1/")

    def fetch(prompt):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"{prompt} Output ONLY the Python code, no explanation."}],
                max_tokens=8192,
                temperature=0.7,
            )
            choice = r.choices[0]
            if choice.finish_reason == "length":
                raise TruncationError(
                    f"Response truncated at {len(choice.message.content)} chars (finish_reason=length)"
                )
            return choice.message.content
        except (RateLimitError, TruncationError):
            raise
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                raise RateLimitError(str(e))
            raise

    return fetch


# --- Core Collection Loop ---

def collect_for_identity(
    identity: str,
    model_name: str,
    fetch_fn,
    output_dir: Path,
    seen_hashes: set[str],
    target: int = TARGET_PER_MODEL,
    dry_run: bool = False,
    summary_sink: Optional[dict] = None,
):
    """
    Collect AI code samples for one model identity.

    Idempotent: checks existing files by index, only fills gaps up to `target`.
    Saves each file immediately with a .json sidecar. Updates seen_hashes in-place.

    Returns the number of newly collected samples.
    """
    # Determine which indices already exist on disk
    existing_indices = set()
    for f in output_dir.glob(f"ai_{identity}_*.py"):
        try:
            existing_indices.add(int(f.stem.rsplit("_", 1)[-1]))
        except ValueError:
            pass

    current_count = len(existing_indices)

    if current_count >= target:
        print(f"  {identity}: already have {current_count}/{target}, skipping.")
        return 0

    needed = target - current_count
    print(f"  {identity}: have {current_count}/{target}, collecting {needed} more...")

    collected = 0
    planned = 0  # for dry-run
    skipped = {"too_short": 0, "invalid_python": 0, "truncated": 0, "duplicate": 0, "error": 0, "rate_limited": 0}
    rejected_dir = output_dir / "rejected"

    for idx in range(target):
        if collected >= needed:
            break
        if idx in existing_indices:
            continue

        prompt = PROMPTS[idx % len(PROMPTS)]

        try:
            raw = fetch_fn(prompt)
        except TruncationError as e:
            print(f"    [{identity}] idx {idx}: TRUNCATED — {e}")
            skipped["truncated"] += 1
            time.sleep(1)
            continue
        except RateLimitError as e:
            print(f"    [{identity}] idx {idx}: RATE-LIMIT — {e}")
            skipped["rate_limited"] += 1
            time.sleep(2)
            continue
        except Exception as e:
            print(f"    [{identity}] idx {idx}: API error — {e}")
            skipped["error"] += 1
            time.sleep(1)
            continue

        code = clean_code(raw)

        if len(code.splitlines()) < MIN_LINES:
            print(f"    [{identity}] idx {idx}: too short ({len(code.splitlines())} lines), skipping")
            skipped["too_short"] += 1
            continue

        syntax_err = _python_syntax_error(code)
        if syntax_err is not None:
            print(
                f"    [{identity}] idx {idx}: invalid Python — "
                f"{syntax_err.msg} at line {syntax_err.lineno}"
            )
            skipped["invalid_python"] += 1
            if not dry_run:
                rejected_dir.mkdir(exist_ok=True)
                rejected_file = rejected_dir / f"ai_{identity}_{idx}.py"
                rejected_file.write_text(code)
                (rejected_dir / f"ai_{identity}_{idx}.py.json").write_text(json.dumps({
                    "identity": identity,
                    "model": model_name,
                    "prompt_index": idx,
                    "prompt": prompt,
                    "rejected_at": int(time.time()),
                    "reason": "invalid_python",
                    "error": str(syntax_err),
                    "error_msg": syntax_err.msg,
                    "error_lineno": syntax_err.lineno,
                    "total_lines": len(code.splitlines()),
                }, indent=2))
            continue

        content_hash = _content_hash(code)
        if content_hash in seen_hashes:
            print(f"    [{identity}] idx {idx}: duplicate content, skipping")
            skipped["duplicate"] += 1
            continue

        # Save sample unless dry-run
        out_file = output_dir / f"ai_{identity}_{idx}.py"
        sidecar = {
            "identity": identity,
            "model": model_name,
            "prompt_index": idx,
            "prompt": prompt,
            "collected_at": int(time.time()),
            "content_hash": content_hash,
            "total_lines": len(code.splitlines()),
            "code_lines": _count_code_lines(code),
        }

        if dry_run:
            planned += 1
            print(
                f"    [{identity}] DRY-RUN would write {out_file.name} "
                f"({len(code.splitlines())} lines)"
            )
        else:
            out_file.write_text(code)
            (output_dir / f"ai_{identity}_{idx}.py.json").write_text(json.dumps(sidecar, indent=2))
            seen_hashes.add(content_hash)
            collected += 1
            print(
                f"    [{identity}] {current_count + collected}/{target}: "
                f"{len(code.splitlines())} lines — {prompt[:55]}..."
            )

            time.sleep(0.5)

    # Summary for this identity
    skip_parts = [f"{v} {k}" for k, v in skipped.items() if v]
    skip_str = f" (skipped: {', '.join(skip_parts)})" if skip_parts else ""
    if dry_run:
        print(f"  {identity}: DRY-RUN +{planned} planned{skip_str}  [total on disk unchanged: {current_count}]")
    else:
        print(f"  {identity}: +{collected} new samples{skip_str}  [total on disk: {current_count + collected}]")

    # Emit per-identity summary to sink
    summary = {
        "identity": identity,
        "model": model_name,
        "existing": current_count,
        "target": target,
        "collected": collected,
        "planned": planned,
        "skipped": skipped,
        "final_on_disk": current_count if dry_run else current_count + collected,
    }
    if summary_sink is not None:
        summary_sink[identity] = summary

    return collected if not dry_run else 0


# --- Main ---

def _parse_enabled_providers() -> Optional[set[str]]:
    """Parse AI_PROVIDERS env var into a normalized set of identities.

    Accepts comma-separated values. Supported tokens (case-insensitive):
      - openai, gpt4o -> gpt4o
      - anthropic, claude -> claude
      - gemini -> gemini
      - deepseek, deepseek_v3 -> deepseek_v3
      - kimi, opencode -> kimi
    Returns None if not set (meaning: do not filter by provider list).
    """
    raw = os.environ.get("AI_PROVIDERS")
    if not raw:
        return None
    mapping = {
        "openai": "gpt4o",
        "gpt4o": "gpt4o",
        "anthropic": "claude",
        "claude": "claude",
        "gemini": "gemini",
        "deepseek": "deepseek_v3",
        "deepseek_v3": "deepseek_v3",
        "kimi": "kimi",
        "opencode": "kimi",
    }
    enabled = set()
    for tok in raw.split(','):
        key = tok.strip().lower()
        if key in mapping:
            enabled.add(mapping[key])
    return enabled


def main():
    base = Path(__file__).parent
    output_dir = base / "research/data/raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    openai_key    = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    gemini_key    = os.environ.get("GEMINI_API_KEY")
    deepseek_key  = os.environ.get("DEEPSEEK_API_KEY")
    opencode_key  = os.environ.get("OPENCODE_API_KEY")

    dry_run = os.environ.get("AI_DRY_RUN", "0") in {"1", "true", "yes", "on"}
    target_override = os.environ.get("AI_TARGET_PER_MODEL")
    try:
        target_per_model = int(target_override) if target_override else TARGET_PER_MODEL
    except ValueError:
        target_per_model = TARGET_PER_MODEL

    enabled_set = _parse_enabled_providers()

    print("=" * 60)
    print("AI Sample Collection from Real LLM APIs")
    print("=" * 60)
    print(f"OpenAI API:    {'OK' if openai_key else 'NOT SET'}")
    print(f"Anthropic API: {'OK' if anthropic_key else 'NOT SET'}")
    print(f"Gemini API:    {'OK' if gemini_key else 'NOT SET'}")
    print(f"DeepSeek API:  {'OK' if deepseek_key else 'NOT SET'}")
    print(f"OpenCode API:  {'OK' if opencode_key else 'NOT SET'} (Kimi via Zen)")
    print(f"Dry-run:        {'ON' if dry_run else 'OFF'}")
    print(f"Target/model:   {target_per_model}")
    if enabled_set is not None:
        print(f"Providers:      only {', '.join(sorted(enabled_set))}")
    print()

    # Load hashes of all existing AI samples for cross-identity deduplication
    seen_hashes: set[str] = set()
    for f in output_dir.glob("ai_*.py"):
        try:
            seen_hashes.add(_content_hash(f.read_text()))
        except Exception:
            pass
    if seen_hashes:
        print(f"Loaded {len(seen_hashes)} existing content hashes for deduplication.\n")

    # (identity, model_name, fetch_fn)
    model_configs = [
        ("gpt4o",       "gpt-4o",              _make_openai_fetcher(openai_key, "gpt-4o")),
        ("claude",      "claude-sonnet-4-6",    _make_anthropic_fetcher(anthropic_key, "claude-sonnet-4-6")),
        ("gemini",      "gemini-2.5-flash",     _make_gemini_fetcher(gemini_key, "gemini-2.5-flash")),
        ("deepseek_v3", "deepseek-chat",        _make_deepseek_fetcher(deepseek_key, "deepseek-chat")),
        ("kimi",        "kimi-k2",              _make_kimi_fetcher(opencode_key, "kimi-k2")),
    ]

    # Filter by AI_PROVIDERS env var if provided
    if enabled_set is not None:
        model_configs = [cfg for cfg in model_configs if cfg[0] in enabled_set]

    total_new = 0
    per_identity_summary: dict[str, dict] = {}
    for identity, model_name, fetch_fn in model_configs:
        print(f"\nProcessing {identity} ({model_name})...")
        if fetch_fn is None:
            print(f"  Skipping {identity}: API key not set or SDK unavailable.")
            continue
        total_new += collect_for_identity(
            identity,
            model_name,
            fetch_fn,
            output_dir,
            seen_hashes,
            target=target_per_model,
            dry_run=dry_run,
            summary_sink=per_identity_summary,
        )

    print("\n" + "=" * 60)
    print("Collection Complete!")
    print("=" * 60)
    print(f"Newly collected this run: {total_new}")
    print("\nSummary by provider:")
    overall = {"collected": 0, "planned": 0, "skipped": {"too_short": 0, "invalid_python": 0, "truncated": 0, "duplicate": 0, "error": 0, "rate_limited": 0}}
    for identity, _, _ in model_configs:
        count = len(list(output_dir.glob(f"ai_{identity}_*.py")))
        summ = per_identity_summary.get(identity, {})
        collected = summ.get("collected", 0)
        planned = summ.get("planned", 0)
        skipped = summ.get("skipped", {})
        overall["collected"] += collected
        overall["planned"] += planned
        for k in overall["skipped"].keys():
            overall["skipped"][k] += int(skipped.get(k, 0))
        target = summ.get("target", TARGET_PER_MODEL)
        print(
            f"  {identity:15s}: on disk {count:3d} / {target}  | "
            f"collected +{collected}  planned +{planned}  "
            f"skipped {skipped}"
        )
    print("\nOverall:")
    print(
        f"  collected +{overall['collected']}  planned +{overall['planned']}  "
        f"skipped {overall['skipped']}"
    )


if __name__ == "__main__":
    main()
