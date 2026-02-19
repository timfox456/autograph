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
import re
import time
import base64
from pathlib import Path

RATE_LIMIT_PATTERNS = (
    "rate limit",
    "secondary rate limit",
    "abuse",
    "403",
)

DENYLIST_PREFIXES = (
    "docs/", "doc/", "examples/", "example/", "vendor/", "third_party/",
)


def _run_gh(cmd: list[str], max_attempts: int = 5, base_sleep: float = 1.0) -> str:
    """Run a gh command with simple exponential backoff on rate limit errors."""
    attempt = 0
    while True:
        try:
            return subprocess.check_output(cmd, timeout=45).decode()
        except Exception as e:
            msg = str(e).lower()
            attempt += 1
            if attempt >= max_attempts or not any(p in msg for p in RATE_LIMIT_PATTERNS):
                raise
            sleep_s = base_sleep * (2 ** (attempt - 1))
            print(f"Rate limited. Backing off {sleep_s:.1f}s (attempt {attempt}/{max_attempts})...")
            time.sleep(sleep_s)


def get_greenfield_files(repo, author, target_count=50, current_count=0):
    """
    Finds new-file commits by an author and extracts the complete files.
    """
    needed = target_count - current_count
    if needed <= 0:
        return []

    print(f"Searching for greenfield files by {author} in {repo} (need {needed})...")
    
    collected_files = []
    page = 1
    
    while len(collected_files) < needed and page <= 5:
        cmd = [
            "gh", "api",
            f"repos/{repo}/commits?author={author}&per_page=100&page={page}",
            "--jq",
            # Filter: single-parent commits and non-merge subjects to reduce misattribution
            ".[] | select((.parents | length) == 1 and (.commit.message | startswith(\"Merge\") | not)) | .sha",
        ]
        try:
            shas = _run_gh(cmd).splitlines()
        except Exception as e:
            print(f"Error fetching commits for {author} on page {page}: {e}")
            break
            
        if not shas:
            break

        for sha in shas:
            if len(collected_files) >= needed:
                break
                
            # Use the commit API to get the list of files and their statuses
            commit_cmd = ["gh", "api", f"repos/{repo}/commits/{sha}"]
            try:
                commit_json = _run_gh(commit_cmd)
                commit_data = json.loads(commit_json)
                # Sanity: prefer commits where author == committer (either login or email)
                author_login = (commit_data.get("author") or {}).get("login")
                committer_login = (commit_data.get("committer") or {}).get("login")
                ca = (commit_data.get("commit") or {}).get("author") or {}
                cc = (commit_data.get("commit") or {}).get("committer") or {}
                author_email = ca.get("email")
                committer_email = cc.get("email")
                if (author_login and committer_login and author_login != committer_login) or (
                    author_email and committer_email and author_email != committer_email
                ):
                    # Skip if clearly authored by someone else but merged/committed by target
                    continue
                
                for file in commit_data.get("files", []):
                    # Only take files that were newly added and are Python files
                    filename = file.get("filename", "")
                    if file.get("status") == "added" and filename.endswith(".py"):
                        # Skip denylisted paths (vendored/docs/examples, etc.)
                        if any(filename.startswith(p) for p in DENYLIST_PREFIXES):
                            continue
                        # Try to get content from patch, fallback to contents API if patch is missing
                        content = None
                        patch = file.get("patch")
                        if patch:
                            lines = [l[1:] for l in patch.splitlines() if l.startswith("+") and not l.startswith("+++")]
                            content = "\n".join(lines)
                        else:
                            # Patch can be missing for large files, fetch via contents API
                            content_cmd = [
                                "gh",
                                "api",
                                f"repos/{repo}/contents/{filename}?ref={sha}",
                                "--jq",
                                ".content",
                            ]
                            try:
                                b64_content = _run_gh(content_cmd).strip()
                                if b64_content:
                                    content = base64.b64decode(b64_content).decode("utf-8", errors="replace")
                            except Exception as e:
                                print(f"    Failed to fetch content for {filename}: {e}")
                                continue
                        
                        if content and content.count("\n") >= 10:
                            collected_files.append(content)
                            print(f"  Captured whole file: {file.get('filename')} (total: {len(collected_files) + current_count})")
                            if len(collected_files) >= needed:
                                break
            except Exception as e:
                continue
        
        page += 1
            
    return collected_files


def main():
    # 15 Unique Human Authors for Greenfield Collection
    # Support multiple handles/emails per author to improve recall across time
    targets = [
        {"repo": "encode/httpx", "authors": ["tomchristie", "tom@tomchristie.com"], "name": "tom"},
        {"repo": "cookiecutter/cookiecutter", "authors": ["audreyfeldroy", "audreyr"], "name": "audrey"},
        {"repo": "python-attrs/attrs", "authors": ["hynek"], "name": "hynek"},
        {"repo": "pyca/cryptography", "authors": ["alex"] , "name": "alex"},
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
    
    for target in targets:
        # Check current count
        current_files = list(output_dir.glob(f"human_{target['name']}_*.py"))
        current_count = len(current_files)
        
        if current_count >= 50:
            print(f"Skipping {target['name']}, already have {current_count} fragments.")
            continue
            
        needed = 50 - current_count
        collected_count = 0

        # Try multiple handles/emails until quota is met
        author_handles = target.get("authors") or [target.get("author")]
        author_handles = [h for h in author_handles if h]

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
                    print(f"Error fetching commits: {e}")
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

                        # Prefer commits where author == committer (login or email)
                        author_login = (commit_data.get("author") or {}).get("login")
                        committer_login = (commit_data.get("committer") or {}).get("login")
                        ca = (commit_data.get("commit") or {}).get("author") or {}
                        cc = (commit_data.get("commit") or {}).get("committer") or {}
                        author_email = ca.get("email")
                        committer_email = cc.get("email")
                        if (author_login and committer_login and author_login != committer_login) or (
                            author_email and committer_email and author_email != committer_email
                        ):
                            continue

                        for file in commit_data.get("files", []):
                            filename = file.get("filename", "")
                            if file.get("status") == "added" and filename.endswith(".py"):
                                if any(filename.startswith(p) for p in DENYLIST_PREFIXES):
                                    continue

                                content = None
                                source = None
                                patch = file.get("patch")
                                if patch:
                                    lines = [l[1:] for l in patch.splitlines() if l.startswith("+") and not l.startswith("+++")]
                                    content = "\n".join(lines)
                                    source = "patch"
                                else:
                                    content_cmd = [
                                        "gh",
                                        "api",
                                        f"repos/{target['repo']}/contents/{filename}?ref={sha}",
                                        "--jq",
                                        ".content",
                                    ]
                                    try:
                                        b64_content = _run_gh(content_cmd).strip()
                                        if b64_content:
                                            content = base64.b64decode(b64_content).decode("utf-8", errors="replace")
                                            source = "contents_api"
                                    except Exception:
                                        continue

                                if content and content.count("\n") >= 10:
                                    index = current_count + collected_count
                                    out_file = output_dir / f"human_{target['name']}_{index}.py"
                                    with open(out_file, "w") as f:
                                        f.write(content)

                                    # Save provenance sidecar
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
                                    }
                                    with open(str(out_file) + ".json", "w") as meta_f:
                                        json.dump(sidecar, meta_f, indent=2)

                                    collected_count += 1
                                    print(
                                        f"  Saved: {filename} (total: {current_count + collected_count})"
                                    )
                                    if collected_count >= needed:
                                        break
                    except Exception:
                        continue
                page += 1
    
    print("Done collecting greenfield data.")

if __name__ == "__main__":
    main()
