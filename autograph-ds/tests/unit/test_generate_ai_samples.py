"""
Unit tests for generate_ai_samples.py

Covers:
  Utilities  — clean_code, _content_hash, _is_valid_python, _count_code_lines
  Factories  — _make_*_fetcher: return None on missing key/SDK; callable otherwise;
               call the correct endpoint/model when invoked
  Collection — collect_for_identity: idempotency, gap-filling, incremental save,
               sidecar JSON, quality gates (min lines, syntax validation),
               content deduplication, API error recovery, fixed prompt assignment
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from generate_ai_samples import (
    PROMPTS,
    TARGET_PER_MODEL,
    _content_hash,
    _count_code_lines,
    _is_valid_python,
    _make_anthropic_fetcher,
    _make_deepseek_fetcher,
    _make_gemini_fetcher,
    _make_kimi_fetcher,
    _make_openai_fetcher,
    clean_code,
    collect_for_identity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GOOD_CODE = "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n    return a + b"
GOOD_CODE_ALT = "def subtract(a, b):\n    \"\"\"Subtract b from a.\"\"\"\n    return a - b"


def _unique_code(tag):
    """Return syntactically valid 3-line Python with a unique tag to avoid hash collisions."""
    return f"def fn_{tag}():\n    '''Unique sample {tag}.'''\n    return {tag!r}"


# ---------------------------------------------------------------------------
# clean_code
# ---------------------------------------------------------------------------

class TestCleanCode:
    def test_strips_python_fence(self):
        raw = "```python\ndef foo():\n    pass\n```"
        assert clean_code(raw) == "def foo():\n    pass"

    def test_strips_generic_fence(self):
        raw = "```\ndef foo():\n    pass\n```"
        assert clean_code(raw) == "def foo():\n    pass"

    def test_strips_leading_blank_lines(self):
        raw = "\n\ndef foo():\n    pass"
        assert clean_code(raw) == "def foo():\n    pass"

    def test_strips_trailing_blank_lines(self):
        raw = "def foo():\n    pass\n\n"
        assert clean_code(raw) == "def foo():\n    pass"

    def test_preserves_internal_blank_lines(self):
        code = "def foo():\n\n    pass"
        assert clean_code(code) == code

    def test_no_fence_passthrough(self):
        code = "x = 1\ny = 2"
        assert clean_code(code) == code

    def test_fence_with_surrounding_blank_lines(self):
        raw = "\n```python\ndef foo():\n    pass\n```\n"
        assert clean_code(raw) == "def foo():\n    pass"


# ---------------------------------------------------------------------------
# _content_hash
# ---------------------------------------------------------------------------

class TestContentHash:
    def test_deterministic(self):
        code = "def foo(): pass"
        assert _content_hash(code) == _content_hash(code)

    def test_length_is_16(self):
        assert len(_content_hash("anything")) == 16

    def test_different_content_yields_different_hash(self):
        assert _content_hash("def foo(): pass") != _content_hash("def bar(): pass")

    def test_identical_content_yields_same_hash(self):
        a = "def palindrome(s):\n    return s == s[::-1]"
        b = "def palindrome(s):\n    return s == s[::-1]"
        assert _content_hash(a) == _content_hash(b)

    def test_whitespace_sensitive(self):
        # Trailing newline produces a different hash
        assert _content_hash("def foo(): pass") != _content_hash("def foo(): pass\n")


# ---------------------------------------------------------------------------
# _is_valid_python
# ---------------------------------------------------------------------------

class TestIsValidPython:
    def test_valid_function(self):
        assert _is_valid_python("def foo():\n    return 1") is True

    def test_valid_class(self):
        assert _is_valid_python("class Foo:\n    pass") is True

    def test_syntax_error_unclosed_paren(self):
        assert _is_valid_python("def foo(x, y\n    return x\n    # end") is False

    def test_indentation_error(self):
        assert _is_valid_python("def foo():\nreturn 1") is False

    def test_empty_string_is_valid(self):
        assert _is_valid_python("") is True

    def test_expression_is_valid(self):
        assert _is_valid_python("x = 1 + 2") is True

    def test_missing_class_colon(self):
        assert _is_valid_python("class Foo\n    pass\n    x = 1") is False


# ---------------------------------------------------------------------------
# _count_code_lines
# ---------------------------------------------------------------------------

class TestCountCodeLines:
    def test_counts_non_blank_non_comment_lines(self):
        code = "def foo():\n    # comment\n    return 1"
        assert _count_code_lines(code) == 2  # def + return, not comment

    def test_blank_lines_excluded(self):
        assert _count_code_lines("x = 1\n\n\ny = 2") == 2

    def test_comment_lines_excluded(self):
        assert _count_code_lines("# comment\nx = 1") == 1

    def test_all_comments_returns_zero(self):
        assert _count_code_lines("# a\n# b\n# c") == 0

    def test_empty_returns_zero(self):
        assert _count_code_lines("") == 0

    def test_inline_comment_counts_as_code(self):
        # A line with code + inline comment is still a code line
        assert _count_code_lines("x = 1  # inline") == 1

    def test_mixed(self):
        code = "# header\n\ndef foo():\n    x = 1\n    # skip\n    return x\n"
        assert _count_code_lines(code) == 3  # def, x=1, return x


# ---------------------------------------------------------------------------
# Fetcher Factories — availability checks
# ---------------------------------------------------------------------------

class TestFetcherFactories:

    # --- OpenAI ---
    def test_openai_returns_none_without_key(self):
        assert _make_openai_fetcher(None, "gpt-4o") is None

    def test_openai_returns_none_without_sdk(self):
        with patch("generate_ai_samples.OpenAI", None):
            assert _make_openai_fetcher("sk-fake", "gpt-4o") is None

    def test_openai_returns_callable_when_available(self):
        with patch("generate_ai_samples.OpenAI", return_value=MagicMock()):
            fn = _make_openai_fetcher("sk-fake", "gpt-4o")
        assert callable(fn)

    def test_openai_fetcher_calls_chat_completions(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = GOOD_CODE
        with patch("generate_ai_samples.OpenAI", return_value=mock_client):
            fn = _make_openai_fetcher("sk-fake", "gpt-4o")
        result = fn("test prompt")
        mock_client.chat.completions.create.assert_called_once()
        assert result == GOOD_CODE

    # --- Anthropic ---
    def test_anthropic_returns_none_without_key(self):
        assert _make_anthropic_fetcher(None, "claude-sonnet-4-6") is None

    def test_anthropic_returns_none_without_sdk(self):
        with patch("generate_ai_samples.anthropic", None):
            assert _make_anthropic_fetcher("sk-fake", "claude-sonnet-4-6") is None

    def test_anthropic_returns_callable_when_available(self):
        mock_module = MagicMock()
        with patch("generate_ai_samples.anthropic", mock_module):
            fn = _make_anthropic_fetcher("sk-fake", "claude-sonnet-4-6")
        assert callable(fn)

    def test_anthropic_fetcher_calls_messages_create(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value.content[0].text = GOOD_CODE
        mock_module = MagicMock()
        mock_module.Anthropic.return_value = mock_client
        with patch("generate_ai_samples.anthropic", mock_module):
            fn = _make_anthropic_fetcher("sk-fake", "claude-sonnet-4-6")
        result = fn("test prompt")
        mock_client.messages.create.assert_called_once()
        assert result == GOOD_CODE

    # --- Gemini ---
    def test_gemini_returns_none_without_key(self):
        assert _make_gemini_fetcher(None, "gemini-2.5-flash") is None

    def test_gemini_returns_none_without_sdk(self):
        with patch("generate_ai_samples.genai", None):
            assert _make_gemini_fetcher("sk-fake", "gemini-2.5-flash") is None

    def test_gemini_returns_callable_when_available(self):
        mock_genai = MagicMock()
        mock_types = MagicMock()
        with patch("generate_ai_samples.genai", mock_genai), \
             patch("generate_ai_samples.types", mock_types):
            fn = _make_gemini_fetcher("sk-fake", "gemini-2.5-flash")
        assert callable(fn)

    # --- DeepSeek ---
    def test_deepseek_returns_none_without_key(self):
        assert _make_deepseek_fetcher(None, "deepseek-chat") is None

    def test_deepseek_returns_callable_with_key(self):
        fn = _make_deepseek_fetcher("sk-fake", "deepseek-chat")
        assert callable(fn)

    def test_deepseek_fetcher_calls_deepseek_endpoint(self):
        fn = _make_deepseek_fetcher("sk-fake", "deepseek-chat")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": GOOD_CODE}}]}
        mock_resp.raise_for_status.return_value = None
        with patch("generate_ai_samples.requests.post", return_value=mock_resp) as mock_post:
            result = fn("test prompt")
        url = mock_post.call_args[0][0]
        assert "deepseek.com" in url
        assert result == GOOD_CODE

    # --- Kimi (OpenCode Zen) ---
    def test_kimi_returns_none_without_key(self):
        assert _make_kimi_fetcher(None, "kimi-k2") is None

    def test_kimi_returns_none_without_openai_sdk(self):
        with patch("generate_ai_samples.OpenAI", None):
            assert _make_kimi_fetcher("sk-fake", "kimi-k2") is None

    def test_kimi_returns_callable_when_available(self):
        with patch("generate_ai_samples.OpenAI", return_value=MagicMock()):
            fn = _make_kimi_fetcher("sk-fake", "kimi-k2")
        assert callable(fn)

    def test_kimi_fetcher_uses_zen_base_url(self):
        """OpenAI client must be initialised with the OpenCode Zen base URL."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = GOOD_CODE
        with patch("generate_ai_samples.OpenAI", return_value=mock_client) as mock_cls:
            fn = _make_kimi_fetcher("sk-fake", "kimi-k2")
            fn("test prompt")
        _, kwargs = mock_cls.call_args
        assert "opencode.ai" in kwargs["base_url"]

    def test_kimi_fetcher_calls_chat_completions(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = GOOD_CODE
        with patch("generate_ai_samples.OpenAI", return_value=mock_client):
            fn = _make_kimi_fetcher("sk-fake", "kimi-k2")
        result = fn("test prompt")
        mock_client.chat.completions.create.assert_called_once()
        assert result == GOOD_CODE


# ---------------------------------------------------------------------------
# collect_for_identity — core collection loop
# ---------------------------------------------------------------------------

class TestCollectForIdentity:

    # --- Idempotency ---

    def test_skips_when_already_at_target(self, tmp_path):
        for i in range(TARGET_PER_MODEL):
            (tmp_path / f"ai_gpt4o_{i}.py").write_text(_unique_code(i))

        fetch_fn = MagicMock(side_effect=AssertionError("should not be called"))
        result = collect_for_identity("gpt4o", "gpt-4o", fetch_fn, tmp_path, set())

        assert result == 0
        fetch_fn.assert_not_called()

    def test_does_not_overwrite_existing_files(self, tmp_path):
        (tmp_path / "ai_gpt4o_0.py").write_text(GOOD_CODE)
        seen = {_content_hash(GOOD_CODE)}

        fetch_fn = MagicMock(return_value=GOOD_CODE_ALT)
        collect_for_identity("gpt4o", "gpt-4o", fetch_fn, tmp_path, seen, target=2)

        assert (tmp_path / "ai_gpt4o_0.py").read_text() == GOOD_CODE

    # --- Gap filling ---

    def test_collects_from_scratch(self, tmp_path):
        fetch_fn = MagicMock(side_effect=[_unique_code(i) for i in range(3)])
        result = collect_for_identity("gpt4o", "gpt-4o", fetch_fn, tmp_path, set(), target=3)

        assert result == 3
        assert len(list(tmp_path.glob("ai_gpt4o_*.py"))) == 3

    def test_fills_only_missing_index(self, tmp_path):
        (tmp_path / "ai_gpt4o_0.py").write_text(GOOD_CODE)
        (tmp_path / "ai_gpt4o_2.py").write_text(GOOD_CODE_ALT)
        seen = {_content_hash(GOOD_CODE), _content_hash(GOOD_CODE_ALT)}

        fetch_fn = MagicMock(return_value=_unique_code("gap"))
        result = collect_for_identity("gpt4o", "gpt-4o", fetch_fn, tmp_path, seen, target=3)

        assert result == 1
        assert (tmp_path / "ai_gpt4o_1.py").exists()
        assert not (tmp_path / "ai_gpt4o_3.py").exists()

    def test_returns_correct_new_count(self, tmp_path):
        for i in range(2):
            (tmp_path / f"ai_claude_{i}.py").write_text(_unique_code(i))
        seen = {_content_hash(_unique_code(i)) for i in range(2)}

        fetch_fn = MagicMock(side_effect=[_unique_code(f"new{i}") for i in range(4)])
        result = collect_for_identity("claude", "claude-sonnet-4-6", fetch_fn, tmp_path, seen, target=4)

        assert result == 2

    # --- Incremental save ---

    def test_each_file_saved_before_next_fetch(self, tmp_path):
        """File for index N exists on disk by the time fetch(N+1) is called."""
        files_at_call = []

        def fetch_fn(prompt):
            files_at_call.append(len(list(tmp_path.glob("ai_gpt4o_*.py"))))
            idx = len(files_at_call) - 1
            return _unique_code(f"call{idx}")

        collect_for_identity("gpt4o", "gpt-4o", fetch_fn, tmp_path, set(), target=3)

        # At call 0: 0 files; call 1: 1 file; call 2: 2 files
        assert files_at_call == [0, 1, 2]

    # --- Sidecar JSON ---

    def test_sidecar_written_alongside_code_file(self, tmp_path):
        fetch_fn = MagicMock(return_value=GOOD_CODE)
        collect_for_identity("gpt4o", "gpt-4o", fetch_fn, tmp_path, set(), target=1)

        assert (tmp_path / "ai_gpt4o_0.py.json").exists()

    def test_sidecar_contains_required_fields(self, tmp_path):
        fetch_fn = MagicMock(return_value=GOOD_CODE)
        collect_for_identity("gpt4o", "gpt-4o", fetch_fn, tmp_path, set(), target=1)

        meta = json.loads((tmp_path / "ai_gpt4o_0.py.json").read_text())
        assert meta["identity"] == "gpt4o"
        assert meta["model"] == "gpt-4o"
        assert meta["prompt_index"] == 0
        assert meta["prompt"] == PROMPTS[0]
        assert "content_hash" in meta
        assert "collected_at" in meta
        assert "total_lines" in meta
        assert "code_lines" in meta

    def test_sidecar_hash_matches_saved_file(self, tmp_path):
        fetch_fn = MagicMock(return_value=GOOD_CODE)
        collect_for_identity("gpt4o", "gpt-4o", fetch_fn, tmp_path, set(), target=1)

        code_on_disk = (tmp_path / "ai_gpt4o_0.py").read_text()
        meta = json.loads((tmp_path / "ai_gpt4o_0.py.json").read_text())
        assert meta["content_hash"] == _content_hash(code_on_disk)

    def test_sidecar_prompt_matches_index(self, tmp_path):
        """Sidecar records the prompt that was actually used (PROMPTS[idx])."""
        fetch_fn = MagicMock(side_effect=[_unique_code(i) for i in range(3)])
        collect_for_identity("gpt4o", "gpt-4o", fetch_fn, tmp_path, set(), target=3)

        for idx in range(3):
            meta = json.loads((tmp_path / f"ai_gpt4o_{idx}.py.json").read_text())
            assert meta["prompt"] == PROMPTS[idx]
            assert meta["prompt_index"] == idx

    # --- Quality gates ---

    def test_skips_too_short_response(self, tmp_path):
        short = "x = 1"  # 1 line, below MIN_LINES=3
        fetch_fn = MagicMock(return_value=short)
        result = collect_for_identity("gpt4o", "gpt-4o", fetch_fn, tmp_path, set(), target=1)

        assert result == 0
        assert not list(tmp_path.glob("ai_gpt4o_*.py"))

    def test_skips_invalid_python(self, tmp_path):
        # 3 lines but syntactically broken (unclosed parenthesis)
        bad = "def foo(x, y\n    return x\n    # end"
        fetch_fn = MagicMock(return_value=bad)
        result = collect_for_identity("gpt4o", "gpt-4o", fetch_fn, tmp_path, set(), target=1)

        assert result == 0
        assert not list(tmp_path.glob("ai_gpt4o_*.py"))

    def test_cleans_markdown_fences_before_saving(self, tmp_path):
        fenced = f"```python\n{GOOD_CODE}\n```"
        fetch_fn = MagicMock(return_value=fenced)
        collect_for_identity("gpt4o", "gpt-4o", fetch_fn, tmp_path, set(), target=1)

        saved = (tmp_path / "ai_gpt4o_0.py").read_text()
        assert "```" not in saved
        assert saved == GOOD_CODE

    # --- Content deduplication ---

    def test_skips_hash_already_in_seen(self, tmp_path):
        seen = {_content_hash(GOOD_CODE)}
        fetch_fn = MagicMock(return_value=GOOD_CODE)
        result = collect_for_identity("gpt4o", "gpt-4o", fetch_fn, tmp_path, seen, target=1)

        assert result == 0
        assert not list(tmp_path.glob("ai_gpt4o_*.py"))

    def test_new_hash_added_to_seen_after_collect(self, tmp_path):
        seen: set = set()
        fetch_fn = MagicMock(return_value=GOOD_CODE)
        collect_for_identity("gpt4o", "gpt-4o", fetch_fn, tmp_path, seen, target=1)

        assert _content_hash(GOOD_CODE) in seen

    def test_cross_identity_dedup_via_seen_hashes(self, tmp_path):
        """A hash pre-populated from a different identity is also blocked."""
        seen = {_content_hash(GOOD_CODE)}  # e.g. already seen from claude
        fetch_fn = MagicMock(return_value=GOOD_CODE)
        result = collect_for_identity("gpt4o", "gpt-4o", fetch_fn, tmp_path, seen, target=1)

        assert result == 0

    # --- Error recovery ---

    def test_api_error_skips_index_and_continues(self, tmp_path):
        side_effects = [
            ConnectionError("timeout"),   # idx=0 fails
            _unique_code("idx1"),         # idx=1 succeeds
            _unique_code("idx2"),         # idx=2 succeeds
        ]
        fetch_fn = MagicMock(side_effect=side_effects)
        result = collect_for_identity("gpt4o", "gpt-4o", fetch_fn, tmp_path, set(), target=3)

        assert result == 2
        assert not (tmp_path / "ai_gpt4o_0.py").exists()
        assert (tmp_path / "ai_gpt4o_1.py").exists()
        assert (tmp_path / "ai_gpt4o_2.py").exists()

    def test_multiple_errors_still_collects_successes(self, tmp_path):
        side_effects = [
            RuntimeError("err"),
            RuntimeError("err"),
            _unique_code("ok"),
        ]
        fetch_fn = MagicMock(side_effect=side_effects)
        result = collect_for_identity("gpt4o", "gpt-4o", fetch_fn, tmp_path, set(), target=3)

        assert result == 1

    # --- Fixed prompt assignment ---

    def test_prompt_assigned_by_index(self, tmp_path):
        received: list[str] = []

        def fetch_fn(prompt):
            received.append(prompt)
            return _unique_code(len(received))

        collect_for_identity("gpt4o", "gpt-4o", fetch_fn, tmp_path, set(), target=3)

        assert received[0] == PROMPTS[0]
        assert received[1] == PROMPTS[1]
        assert received[2] == PROMPTS[2]

    def test_prompt_assignment_consistent_across_runs(self, tmp_path):
        """Re-running with a gap fills the missing index with the same prompt."""
        # First run: collect indices 0 and 2
        (tmp_path / "ai_gpt4o_0.py").write_text(GOOD_CODE)
        (tmp_path / "ai_gpt4o_2.py").write_text(GOOD_CODE_ALT)
        seen = {_content_hash(GOOD_CODE), _content_hash(GOOD_CODE_ALT)}

        received: list[str] = []

        def fetch_fn(prompt):
            received.append(prompt)
            return _unique_code("gap")

        collect_for_identity("gpt4o", "gpt-4o", fetch_fn, tmp_path, seen, target=3)

        # Index 1 was missing; it should receive PROMPTS[1]
        assert received == [PROMPTS[1]]

    def test_prompts_wrap_past_end_of_list(self, tmp_path):
        """When target > len(PROMPTS), indices wrap via modulo."""
        target = len(PROMPTS) + 2
        received: list[str] = []

        def fetch_fn(prompt):
            received.append(prompt)
            return _unique_code(len(received))

        collect_for_identity("gpt4o", "gpt-4o", fetch_fn, tmp_path, set(), target=target)

        assert received[len(PROMPTS)]     == PROMPTS[0]  # wraps
        assert received[len(PROMPTS) + 1] == PROMPTS[1]
