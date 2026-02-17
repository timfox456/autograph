import urllib.request
import os

def download_sample(url, filename):
    print(f"Downloading {url} to {filename}...")
    with urllib.request.urlopen(url) as response:
        content = response.read().decode('utf-8')
        with open(filename, 'w') as f:
            f.write(content)

samples = {
    "human_requests.py": "https://raw.githubusercontent.com/psf/requests/main/src/requests/sessions.py",
    "human_django.py": "https://raw.githubusercontent.com/django/django/main/django/db/models/base.py",
    "human_fastapi.py": "https://raw.githubusercontent.com/tiangolo/fastapi/master/fastapi/applications.py"
}

os.makedirs("autograph-ds/research/data/raw", exist_ok=True)

for filename, url in samples.items():
    download_sample(url, f"autograph-ds/research/data/raw/{filename}")
