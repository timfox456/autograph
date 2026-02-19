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
from pathlib import Path

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
            "--jq", ".[] | .sha"
        ]
        try:
            shas = subprocess.check_output(cmd, timeout=30).decode().splitlines()
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
                commit_json = subprocess.check_output(commit_cmd, timeout=30).decode()
                commit_data = json.loads(commit_json)
                
                for file in commit_data.get("files", []):
                    # Only take files that were newly added and are Python files
                    if file.get("status") == "added" and file.get("filename", "").endswith(".py"):
                        # Try to get content from patch, fallback to contents API if patch is missing
                        content = None
                        patch = file.get("patch")
                        if patch:
                            lines = [l[1:] for l in patch.splitlines() if l.startswith("+") and not l.startswith("+++")]
                            content = "\n".join(lines)
                        else:
                            # Patch can be missing for large files, fetch via contents API
                            filename = file.get("filename")
                            content_cmd = ["gh", "api", f"repos/{repo}/contents/{filename}?ref={sha}", "--jq", ".content"]
                            try:
                                b64_content = subprocess.check_output(content_cmd, timeout=30).decode().strip()
                                if b64_content:
                                    import base64
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
    # 15 Human Authors for Greenfield Collection
    # 15 Unique Human Authors for Greenfield Collection
    targets = [
        {"repo": "encode/httpx", "author": "tom@tomchristie.com", "name": "tom"},
        {"repo": "cookiecutter/cookiecutter", "author": "audreyfeldroy", "name": "audrey"},
        {"repo": "python-attrs/attrs", "author": "hynek", "name": "hynek"},
        {"repo": "pyca/cryptography", "author": "alex", "name": "alex"},
        {"repo": "django/django", "author": "felixxm", "name": "mariusz"},
        {"repo": "psf/requests", "author": "kennethreitz", "name": "kenneth"},
        {"repo": "pallets/flask", "author": "davidism", "name": "david"},
        {"repo": "tiangolo/fastapi", "author": "tiangolo", "name": "sebastian"},
        {"repo": "psf/black", "author": "ambv", "name": "lukasz"},
        {"repo": "pallets/click", "author": "mitsuhiko", "name": "armin"},
        {"repo": "pypa/warehouse", "author": "dstufft", "name": "donald"},
        {"repo": "pydantic/pydantic", "author": "samuelcolvin", "name": "samuel"},
        {"repo": "Textualize/rich", "author": "willmcgugan", "name": "will"},
        {"repo": "twisted/twisted", "author": "glyph", "name": "glyph"},
        {"repo": "sigmavirus24/github3.py", "author": "sigmavirus24", "name": "ian"}
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
        print(f"Searching for greenfield files by {target['author']} in {target['repo']} (need {needed})...")
        
        collected_count = 0
        page = 1
        while collected_count < needed and page <= 5:
            cmd = [
                "gh", "api", 
                f"repos/{target['repo']}/commits?author={target['author']}&per_page=100&page={page}",
                "--jq", ".[] | .sha"
            ]
            try:
                shas = subprocess.check_output(cmd, timeout=30).decode().splitlines()
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
                    commit_json = subprocess.check_output(commit_cmd, timeout=30).decode()
                    commit_data = json.loads(commit_json)
                    for file in commit_data.get("files", []):
                        if file.get("status") == "added" and file.get("filename", "").endswith(".py"):
                            content = None
                            patch = file.get("patch")
                            if patch:
                                lines = [l[1:] for l in patch.splitlines() if l.startswith("+") and not l.startswith("+++")]
                                content = "\n".join(lines)
                            else:
                                filename = file.get("filename")
                                content_cmd = ["gh", "api", f"repos/{target['repo']}/contents/{filename}?ref={sha}", "--jq", ".content"]
                                try:
                                    b64_content = subprocess.check_output(content_cmd, timeout=30).decode().strip()
                                    if b64_content:
                                        import base64
                                        content = base64.b64decode(b64_content).decode("utf-8", errors="replace")
                                except:
                                    continue
                            
                            if content and content.count("\n") >= 10:
                                index = current_count + collected_count
                                out_file = output_dir / f"human_{target['name']}_{index}.py"
                                with open(out_file, "w") as f:
                                    f.write(content)
                                collected_count += 1
                                print(f"  Captured and saved: {file.get('filename')} (total: {current_count + collected_count})")
                                if collected_count >= needed:
                                    break
                except Exception:
                    continue
            page += 1
    
    print("Done collecting greenfield data.")

if __name__ == "__main__":
    main()
