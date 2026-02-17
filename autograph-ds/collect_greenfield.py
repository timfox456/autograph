import subprocess
import json
import os

def get_greenfield_fragments(repo, author, limit=5):
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
        shas = subprocess.check_output(cmd).decode().splitlines()
    except Exception as e:
        print(f"Error fetching commits: {e}")
        return []

    fragments = []
    for sha in shas:
        # 2. Get the diff for the commit
        diff_cmd = ["gh", "api", f"repos/{repo}/commits/{sha}", "--header", "Accept: application/vnd.github.v3.diff"]
        try:
            diff = subprocess.check_output(diff_cmd).decode()
            # 3. Filter for added lines that look like new Python logic
            added_lines = []
            for line in diff.splitlines():
                if line.startswith("+") and not line.startswith("+++"):
                    clean_line = line[1:]
                    if clean_line.strip():
                        added_lines.append(clean_line)
            
            if added_lines:
                fragments.append("\n".join(added_lines))
        except:
            continue
            
    return fragments

def main():
    # Example: Mariusz Felisiak (Django) and Kenneth Reitz (Requests - legacy)
    targets = [
        {"repo": "django/django", "author": "felixxm", "name": "mariusz"},
        {"repo": "psf/requests", "author": "kennethreitz", "name": "kenneth"}
    ]
    
    output_dir = "autograph-ds/research/data/raw"
    os.makedirs(output_dir, exist_ok=True)
    
    for target in targets:
        code_blocks = get_greenfield_fragments(target["repo"], target["author"])
        for i, block in enumerate(code_blocks):
            filename = f"human_{target['name']}_{i}.py"
            with open(os.path.join(output_dir, filename), "w") as f:
                f.write(block)
            print(f"Saved greenfield fragment: {filename}")

if __name__ == "__main__":
    main()
