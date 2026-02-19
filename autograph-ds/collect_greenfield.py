"""
Greenfield Code Collection Script

This script collects "greenfield" code samples (newly added code) from GitHub repositories
using the GitHub CLI (gh). It extracts code from commits by specific authors to build
a training dataset of human-authored code with known provenance.

Requirements:
    - GitHub CLI (gh) must be installed and authenticated
    - Run: gh auth login

Usage:
    python collect_greenfield.py

The script will download code samples from configured repositories and save them
to research/data/raw/ (relative to this script).
"""
import subprocess
import json
import os
import time
import base64
import hashlib
from pathlib import Path
from collections import defaultdict

RATE_LIMIT_PATTERNS = (
    "rate limit",
    "secondary rate limit",
    "abuse",
    "403",
)

DENYLIST_PREFIXES = (
    "docs/", "doc/", "examples/", "example/", "vendor/", "third_party/",
    "tests/", "test/", "testing/", "scripts/", "tools/", "bin/", "ci/",
    "benchmarks/", "benchmark/", "profiling/", "stubs/",
)

DENYLIST_FILES = frozenset({
    "setup.py", "conftest.py", "manage.py", "wsgi.py", "asgi.py",
    "conf.py", "config.py", "settings.py", "urls.py", "celery.py",
    "fabfile.py", "tasks.py", "tox.py", "versioneer.py",
    "_compat.py", "compat.py", "six.py", "_six.py",
    "__about__.py", "__version__.py", "_version.py",
})

MIN_TOTAL_LINES = 5
MIN_CODE_LINES = 3


def _run_gh(cmd: list[str], max_attempts: int = 5, base_sleep: float = 1.0) -> str:
    """Run a gh command with simple exponential backoff on rate limit errors."""
    attempt = 0
    while True:
        try:
            return subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=45).decode()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            output = getattr(e, "output", b"").decode().lower()
            msg = str(e).lower() + " " + output
            attempt += 1
            if attempt >= max_attempts or not any(p in msg for p in RATE_LIMIT_PATTERNS):
                raise
            sleep_s = base_sleep * (2 ** (attempt - 1))
            print(f"Rate limited. Backing off {sleep_s:.1f}s (attempt {attempt}/{max_attempts})...")
            time.sleep(sleep_s)


def _is_misattributed(commit_data: dict) -> bool:
    """
    Check if the commit appears to be authored by someone else but merged/committed by the target.
    Sanity: prefer commits where author == committer (either login or email).
    """
    author_login = (commit_data.get("author") or {}).get("login")
    committer_login = (commit_data.get("committer") or {}).get("login")
    ca = (commit_data.get("commit") or {}).get("author") or {}
    cc = (commit_data.get("commit") or {}).get("committer") or {}
    author_email = ca.get("email")
    committer_email = cc.get("email")

    if (author_login and committer_login and author_login != committer_login) or (
        author_email and committer_email and author_email != committer_email
    ):
        return True
    return False


def _is_valid_python(code: str) -> bool:
    """Check if code is syntactically valid Python."""
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def _count_code_lines(content: str) -> int:
    """Count non-blank, non-comment lines of code."""
    code_lines = 0
    for line in content.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            code_lines += 1
    return code_lines


def _content_hash(content: str) -> str:
    """Generate a hash for deduplication."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _is_denylisted(filename: str) -> bool:
    """Check if filename matches denylist patterns."""
    basename = os.path.basename(filename)
    if basename in DENYLIST_FILES:
        return True
    if any(filename.startswith(p) or (p in filename) for p in DENYLIST_PREFIXES):
        return True
    return False


def main():
    targets = [
        {"repo": "encode/httpx", "authors": ["tomchristie", "tom@tomchristie.com"], "name": "tom"},
        {"repo": "cookiecutter/cookiecutter", "authors": ["audreyfeldroy", "audreyr"], "name": "audrey"},
        {"repo": "python-attrs/attrs", "authors": ["hynek"], "name": "hynek"},
        {"repo": "pyca/cryptography", "authors": ["alex"], "name": "alex"},
        {"repo": "django/django", "authors": ["felixxm"], "name": "mariusz"},
        {"repo": "psf/requests", "authors": ["kennethreitz"], "name": "kenneth"},
        {"repo": "pallets/flask", "authors": ["davidism"], "name": "david"},
        {"repo": "tiangolo/fastapi", "authors": ["tiangolo"], "name": "sebastian"},
        {"repo": "psf/black", "authors": ["ambv"], "name": "lukasz"},
        {"repo": "pallets/click", "authors": ["mitsuhiko"], "name": "armin"},
        {"repo": "pypa/warehouse", "authors": ["dstufft"], "name": "donald"},
        {"repo": "pydantic/pydantic", "authors": ["samuelcolvin"], "name": "samuel"},
        {"repo": "Textualize/rich", "authors": ["willmcgugan"], "name": "will"},
        {"repo": "twisted/twisted", "authors": ["glyph"], "name": "glyph"},
        {"repo": "sigmavirus24/github3.py", "authors": ["sigmavirus24"], "name": "ian"},
    ]

    base = Path(__file__).parent
    output_dir = base / "research/data/raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes = set()
    for existing_file in output_dir.glob("human_*.py"):
        try:
            content = existing_file.read_text()
            seen_hashes.add(_content_hash(content))
        except Exception:
            pass

    stats = defaultdict(lambda: defaultdict(int))

    for target in targets:
        author_name = target["name"]
        current_files = list(output_dir.glob(f"human_{author_name}_*.py"))
        current_count = len(current_files)

        if current_count >= 50:
            print(f"Skipping {author_name}, already have {current_count} fragments.")
            continue

        needed = 50 - current_count
        collected_count = 0
        author_hashes = set()

        author_handles = [h for h in target["authors"] if h]

        for handle in author_handles:
            if collected_count >= needed:
                break

            print(f"Searching for greenfield files by {handle} in {target['repo']} (need {needed - collected_count})...")

            page = 1
            while collected_count < needed and page <= 5:
                cmd = [
                    "gh", "api",
                    f"repos/{target['repo']}/commits?author={handle}&per_page=100&page={page}",
                    "--jq",
                    ".[] | select((.parents | length) == 1 and (.commit.message | startswith(\"Merge\") | not)) | .sha",
                ]
                try:
                    shas = _run_gh(cmd).splitlines()
                except Exception as e:
                    print(f"Error fetching commits for {handle} on page {page}: {e}")
                    break

                if not shas:
                    break

                for sha in shas:
                    if collected_count >= needed:
                        break

                    commit_cmd = ["gh", "api", f"repos/{target['repo']}/commits/{sha}"]
                    try:
                        commit_json = _run_gh(commit_cmd)
                        commit_data = json.loads(commit_json)

                        if _is_misattributed(commit_data):
                            stats[author_name]["misattributed"] += 1
                            continue

                        author_login = (commit_data.get("author") or {}).get("login")
                        ca = (commit_data.get("commit") or {}).get("author") or {}
                        cc = (commit_data.get("commit") or {}).get("committer") or {}
                        author_email = ca.get("email")
                        committer_login = (commit_data.get("committer") or {}).get("login")
                        committer_email = cc.get("email")

                        for file in commit_data.get("files", []):
                            filename = file.get("filename", "")

                            if file.get("status") != "added" or not filename.endswith(".py"):
                                continue

                            if _is_denylisted(filename):
                                stats[author_name]["denylisted"] += 1
                                continue

                            content = None
                            source = None
                            patch = file.get("patch")

                            if patch:
                                lines = [raw[1:] for raw in patch.splitlines() if raw.startswith("+") and not raw.startswith("+++")]
                                content = "\n".join(lines)
                                source = "patch"
                            else:
                                content_cmd = [
                                    "gh", "api",
                                    f"repos/{target['repo']}/contents/{filename}?ref={sha}",
                                    "--jq", ".content",
                                ]
                                try:
                                    b64_content = _run_gh(content_cmd).strip()
                                    if b64_content:
                                        content = base64.b64decode(b64_content).decode("utf-8", errors="replace")
                                        source = "contents_api"
                                except Exception as e:
                                    print(f"    Failed to fetch content for {filename} at {sha}: {e}")
                                    stats[author_name]["fetch_failed"] += 1
                                    continue

                            if not content:
                                stats[author_name]["empty_content"] += 1
                                continue

                            total_lines = len(content.splitlines())
                            if total_lines < MIN_TOTAL_LINES:
                                stats[author_name]["too_short"] += 1
                                continue

                            code_lines = _count_code_lines(content)
                            if code_lines < MIN_CODE_LINES:
                                stats[author_name]["too_few_code_lines"] += 1
                                continue

                            if not _is_valid_python(content):
                                stats[author_name]["invalid_python"] += 1
                                print(f"  Skipping {filename}: invalid Python syntax")
                                continue

                            content_hash = _content_hash(content)
                            if content_hash in seen_hashes or content_hash in author_hashes:
                                stats[author_name]["duplicate"] += 1
                                continue

                            index = current_count + collected_count
                            out_file = output_dir / f"human_{author_name}_{index}.py"
                            with open(out_file, "w") as f:
                                f.write(content)

                            sidecar = {
                                "repo": target["repo"],
                                "sha": sha,
                                "filename": filename,
                                "status": file.get("status"),
                                "additions": file.get("additions"),
                                "deletions": file.get("deletions"),
                                "changes": file.get("changes"),
                                "author_login": author_login,
                                "author_email": author_email,
                                "committer_login": committer_login,
                                "committer_email": committer_email,
                                "parents_count": len(commit_data.get("parents", [])),
                                "collected_via": source,
                                "captured_at": int(time.time()),
                                "content_hash": content_hash,
                                "total_lines": total_lines,
                                "code_lines": code_lines,
                            }
                            with open(str(out_file) + ".json", "w") as meta_f:
                                json.dump(sidecar, meta_f, indent=2)

                            author_hashes.add(content_hash)
                            seen_hashes.add(content_hash)
                            collected_count += 1
                            stats[author_name]["collected"] += 1
                            print(f"  Saved: {filename} ({code_lines} code lines, total: {current_count + collected_count})")

                            if collected_count >= needed:
                                break
                    except Exception as e:
                        print(f"  Skipping {sha}: {e}")
                        stats[author_name]["errors"] += 1
                        continue
                page += 1

    print("\n" + "=" * 60)
    print("Collection Summary")
    print("=" * 60)
    for author in sorted(stats.keys()):
        author_stats = stats[author]
        print(f"\n{author}:")
        print(f"  Collected: {author_stats.get('collected', 0)}")
        if author_stats.get("misattributed"):
            print(f"  Skipped (misattributed): {author_stats['misattributed']}")
        if author_stats.get("denylisted"):
            print(f"  Skipped (denylisted path): {author_stats['denylisted']}")
        if author_stats.get("too_short"):
            print(f"  Skipped (too short): {author_stats['too_short']}")
        if author_stats.get("too_few_code_lines"):
            print(f"  Skipped (too few code lines): {author_stats['too_few_code_lines']}")
        if author_stats.get("invalid_python"):
            print(f"  Skipped (invalid Python): {author_stats['invalid_python']}")
        if author_stats.get("duplicate"):
            print(f"  Skipped (duplicate): {author_stats['duplicate']}")
        if author_stats.get("fetch_failed"):
            print(f"  Skipped (fetch failed): {author_stats['fetch_failed']}")
        if author_stats.get("errors"):
            print(f"  Errors: {author_stats['errors']}")

    print("\nDone collecting greenfield data.")


if __name__ == "__main__":
    main()
