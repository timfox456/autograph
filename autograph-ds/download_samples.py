"""
Sample Code Downloader

This script downloads Python source files from well-known open-source projects
to use as training data for human-authored code. These files represent distinct
coding styles from different authors and projects.

Usage:
    python download_samples.py

The script will download sample files to autograph-ds/research/data/raw/
"""
import urllib.request
import os

def download_sample(url, filename):
    """
    Download a single source file from a URL.

    Args:
        url: Direct URL to the raw source file
        filename: Local path where the file should be saved
    """
    print(f"Downloading {url} to {filename}...")
    with urllib.request.urlopen(url) as response:
        content = response.read().decode('utf-8')
        with open(filename, 'w') as f:
            f.write(content)

# Sample files from major Python projects with distinct coding styles
samples = {
    "human_requests.py": "https://raw.githubusercontent.com/psf/requests/main/src/requests/sessions.py",
    "human_django.py": "https://raw.githubusercontent.com/django/django/main/django/db/models/base.py",
    "human_fastapi.py": "https://raw.githubusercontent.com/tiangolo/fastapi/master/fastapi/applications.py"
}

os.makedirs("autograph-ds/research/data/raw", exist_ok=True)

for filename, url in samples.items():
    download_sample(url, f"autograph-ds/research/data/raw/{filename}")
