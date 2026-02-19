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
from pathlib import Path

def get_greenfield_fragments(repo, author, limit=100):
    """
    Uses gh to find commits by an author and extract newly added code blocks.
    """
    print(f"Searching for greenfield commits by {author} in {repo}...")
    
    # 1. Get list of commits by author
    cmd = [
        "gh", "api", 
        f"repos/{repo}/commits?author={author}&per_page={limit}",
        "--jq", ".[] | .sha"
    ]
    try:
        shas = subprocess.check_output(cmd, timeout=30).decode().splitlines()
    except Exception as e:
        print(f"Error fetching commits for {author}: {e}")
        return []

    fragments = []
    for sha in shas:
        if len(fragments) >= 50:
            break
            
        # 2. Get the diff for the commit directly (faster than checking file list separately)
        diff_cmd = ["gh", "api", f"repos/{repo}/commits/{sha}", "--header", "Accept: application/vnd.github.v3.diff"]
        try:
            diff = subprocess.check_output(diff_cmd, timeout=30).decode()
            
            # 3. Process diff to extract added Python lines
            added_lines = []
            is_py_file = False
            file_fragments = []
            
            for line in diff.splitlines():
                if line.startswith("+++ b/") and line.endswith(".py"):
                    # Save previous file fragment if any
                    if added_lines and len(added_lines) > 5:
                        file_fragments.append("\n".join(added_lines))
                    added_lines = []
                    is_py_file = True
                    continue
                elif line.startswith("--- a/"):
                    # Save previous file fragment if any
                    if added_lines and len(added_lines) > 5:
                        file_fragments.append("\n".join(added_lines))
                    added_lines = []
                    is_py_file = False
                    continue
                elif line.startswith("@@"):
                    # Don't save on chunk headers, but keep current is_py_file status
                    continue
                
                if is_py_file and line.startswith("+") and not line.startswith("+++"):
                    clean_line = line[1:]
                    if clean_line.strip():
                        added_lines.append(clean_line)
            
            # Final capture for the last file in diff
            if added_lines and len(added_lines) > 5:
                file_fragments.append("\n".join(added_lines))
            
            if file_fragments:
                # Add all valid fragments from this commit
                fragments.extend(file_fragments)
                print(f"  Captured {len(file_fragments)} fragments from commit {sha[:7]}")
        except Exception as e:
            print(f"  Error processing commit {sha[:7]}: {e}")
            continue
            
    return fragments[:50] # Cap at 50 per author as requested

def main():
    # 10 Human Authors for Greenfield Collection
    targets = [
        {"repo": "django/django", "author": "felixxm", "name": "mariusz"},
        {"repo": "psf/requests", "author": "kennethreitz", "name": "kenneth"},
        {"repo": "pallets/flask", "author": "davidism", "name": "david"},
        {"repo": "tiangolo/fastapi", "author": "tiangolo", "name": "sebastian"},
        {"repo": "python-attrs/attrs", "author": "hynek", "name": "hynek"},
        {"repo": "psf/black", "author": "ambv", "name": "lukasz"},
        {"repo": "encode/httpx", "author": "tomchristie", "name": "tom"},
        {"repo": "pallets/click", "author": "mitsuhiko", "name": "armin"},
        {"repo": "pypa/warehouse", "author": "dstufft", "name": "donald"},
        {"repo": "pydantic/pydantic", "author": "samuelcolvin", "name": "samuel"},
        {"repo": "Textualize/rich", "author": "willmcgugan", "name": "will"},
        {"repo": "twisted/twisted", "author": "glyph", "name": "glyph"},
        {"repo": "pyca/cryptography", "author": "alex", "name": "alex"},
        {"repo": "sigmavirus24/github3.py", "author": "sigmavirus24", "name": "ian"}
    ]
    
    base = Path(__file__).parent
    output_dir = base / "research/data/raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for target in targets:
        # Check current count
        current_files = list(output_dir.glob(f"human_{target['name']}_*.py"))
        if len(current_files) >= 50:
            print(f"Skipping {target['name']}, already have {len(current_files)} fragments.")
            continue
            
        code_blocks = get_greenfield_fragments(target["repo"], target["author"], limit=100)
        print(f"Collected {len(code_blocks)} fragments for {target['name']}")
        for i, block in enumerate(code_blocks):
            filename = f"human_{target['name']}_{i}.py"
            with open(output_dir / filename, "w") as f:
                f.write(block)
    
    print("Done collecting greenfield data.")

if __name__ == "__main__":
    main()
