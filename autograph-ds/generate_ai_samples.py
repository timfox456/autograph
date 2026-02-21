"""
Generate AI Code Samples from Real LLM APIs

This script fetches genuine code samples from various AI models using their APIs.
It replaces synthetic stubs with real model outputs for authentic attestation training.

Required environment variables (in .env file):
    OPENAI_API_KEY - For GPT-4o samples
    ANTHROPIC_API_KEY - For Claude 3.5 samples
    GEMINI_API_KEY - For Gemini samples
    OPENCODE_API_KEY - For DeepSeek samples
"""

import os
from pathlib import Path
import random
import time
import requests

# Load .env from project root (parent of autograph-ds)
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Optional: Real LLM Clients
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


# --- Expanded Prompts for More Diverse Samples ---

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
]


def clean_code(code: str) -> str:
    """Clean up code by removing markdown fences and extra whitespace."""
    code = code.replace("```python", "").replace("```", "").strip()
    # Remove leading/trailing blank lines
    lines = code.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def fetch_openai_samples(api_key, model="gpt-4o", count=25):
    """Fetch real samples from OpenAI API."""
    samples = []
    if not api_key or not OpenAI:
        print(f"  Skipping {model}: OpenAI not available")
        return samples
    
    client = OpenAI(api_key=api_key)
    selected_prompts = random.sample(PROMPTS, min(count, len(PROMPTS)))
    
    for i, prompt in enumerate(selected_prompts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"{prompt} Output ONLY the Python code, no explanation."}],
                temperature=0.7,
            )
            code = clean_code(response.choices[0].message.content)
            if len(code.splitlines()) >= 3:
                samples.append(code)
                print(f"  Sample {i+1}/{len(selected_prompts)}: {len(code.splitlines())} lines")
            else:
                print(f"  Sample {i+1}: too short ({len(code.splitlines())} lines)")
            time.sleep(0.5)
        except Exception as e:
            print(f"  Error on sample {i+1}: {e}")
            time.sleep(1)
    
    return samples


def fetch_anthropic_samples(api_key, model="claude-3-5-sonnet-20241022", count=25):
    """Fetch real samples from Anthropic API."""
    samples = []
    if not api_key or not anthropic:
        print(f"  Skipping {model}: Anthropic not available")
        return samples
    
    client = anthropic.Anthropic(api_key=api_key)
    selected_prompts = random.sample(PROMPTS, min(count, len(PROMPTS)))
    
    for i, prompt in enumerate(selected_prompts):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                messages=[{"role": "user", "content": f"{prompt} Output ONLY the Python code, no explanation."}]
            )
            code = clean_code(response.content[0].text)
            if len(code.splitlines()) >= 3:
                samples.append(code)
                print(f"  Sample {i+1}/{len(selected_prompts)}: {len(code.splitlines())} lines")
            else:
                print(f"  Sample {i+1}: too short ({len(code.splitlines())} lines)")
            time.sleep(0.5)
        except Exception as e:
            print(f"  Error on sample {i+1}: {e}")
            time.sleep(1)
    
    return samples


def fetch_gemini_samples(api_key, model="gemini-2.5-flash", count=15):
    """Fetch real samples from Google Gemini API using new google.genai SDK."""
    samples = []
    if not api_key or not genai:
        print(f"  Skipping {model}: Gemini not available")
        return samples
    
    # Initialize client with API key
    client = genai.Client(api_key=api_key)
    
    # Configure for code generation
    config = types.GenerateContentConfig(
        system_instruction="You are an expert Python developer. Provide only high-quality, PEP 8 compliant code without explanations.",
        temperature=0.3,
    )
    
    selected_prompts = random.sample(PROMPTS, min(count, len(PROMPTS)))
    
    for i, prompt in enumerate(selected_prompts):
        try:
            response = client.models.generate_content(
                model=model,
                contents=f"{prompt} Output ONLY the Python code, no explanation.",
                config=config
            )
            code = clean_code(response.text)
            if len(code.splitlines()) >= 3:
                samples.append(code)
                print(f"  Sample {i+1}/{len(selected_prompts)}: {len(code.splitlines())} lines")
            else:
                print(f"  Sample {i+1}: too short ({len(code.splitlines())} lines)")
            time.sleep(0.5)
        except Exception as e:
            print(f"  Error on sample {i+1}: {e}")
            time.sleep(1)
    
    return samples


def fetch_deepseek_samples(api_key, model="deepseek-chat", count=25):
    """Fetch real samples from DeepSeek API."""
    samples = []
    if not api_key:
        print(f"  Skipping {model}: OPENCODE_API_KEY not set")
        return samples
    
    selected_prompts = random.sample(PROMPTS, min(count, len(PROMPTS)))
    
    for i, prompt in enumerate(selected_prompts):
        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": f"{prompt} Output ONLY the Python code, no explanation."}],
                    "temperature": 0.7,
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            code = clean_code(data["choices"][0]["message"]["content"])
            if len(code.splitlines()) >= 3:
                samples.append(code)
                print(f"  Sample {i+1}/{len(selected_prompts)}: {len(code.splitlines())} lines")
            else:
                print(f"  Sample {i+1}: too short ({len(code.splitlines())} lines)")
            time.sleep(0.5)
        except Exception as e:
            print(f"  Error on sample {i+1}: {e}")
            time.sleep(1)
    
    return samples


# --- Main Execution ---

def main():
    base = Path(__file__).parent
    output_dir = base / "research/data/raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get API keys
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    opencode_key = os.environ.get("OPENCODE_API_KEY")
    
    print("=" * 60)
    print("AI Sample Collection from Real LLM APIs")
    print("=" * 60)
    print(f"OpenAI API: {'OK' if openai_key else 'NOT SET'}")
    print(f"Anthropic API: {'OK' if anthropic_key else 'NOT SET'}")
    print(f"Gemini API: {'OK' if gemini_key else 'NOT SET'}")
    print(f"OpenCode API: {'OK' if opencode_key else 'NOT SET'}")
    print()
    
    # Fetch real samples
    samples = {}
    
    print("Fetching GPT-4o samples...")
    samples["gpt4o"] = fetch_openai_samples(openai_key, "gpt-4o", 15)
    print(f"  Total: {len(samples['gpt4o'])} samples\n")
    
    print("Fetching Claude samples...")
    samples["claude"] = fetch_anthropic_samples(anthropic_key, "claude-sonnet-4-6", 15)
    print(f"  Total: {len(samples['claude'])} samples\n")
    
    print("Fetching Gemini samples...")
    samples["gemini"] = fetch_gemini_samples(gemini_key, "gemini-2.5-flash", 15)
    print(f"  Total: {len(samples['gemini'])} samples\n")
    
    print("Fetching DeepSeek V3 samples (SKIPPED - API unavailable)...")
    # Note: DeepSeek API not accessible with current OPENCODE_API_KEY
    # Would need proper DeepSeek API key to use https://api.deepseek.com
    samples["deepseek_v3"] = []
    print(f"  Total: {len(samples['deepseek_v3'])} samples\n")
    
    # Save all samples
    print("=" * 60)
    print("Saving samples...")
    print("=" * 60)
    for model_name, model_samples in samples.items():
        for i, code in enumerate(model_samples):
            filename = f"ai_{model_name}_{i}.py"
            filepath = output_dir / filename
            with open(filepath, "w") as f:
                f.write(code)
        print(f"  {model_name}: {len(model_samples)} samples saved")
    
    print("\n" + "=" * 60)
    print("Collection Complete!")
    print("=" * 60)
    total = sum(len(s) for s in samples.values())
    print(f"Total real AI samples collected: {total}")
    for model_name, model_samples in samples.items():
        print(f"  {model_name:15s}: {len(model_samples):3d} samples")


if __name__ == "__main__":
    main()
