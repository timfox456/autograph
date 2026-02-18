import os
from pathlib import Path
import random
import time
from dotenv import load_dotenv

# Load environment variables from .env if it exists
load_dotenv()

# Optional: Real LLM Clients
try:
    import openai
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# --- Synthetic Generators (Deterministic Fallbacks) ---

def generate_gpt4o_style(task_id):
    """GPT-4o: Verbose, many comments, uses type hints, prefers f-strings."""
    tasks = [
        f"def process_data_{task_id}(items):\n    \"\"\"Processes a list of items and returns the sum.\"\"\"\n    total = 0\n    for item in items:\n        # Add the item to the total\n        total += item\n    return f'The total is {{total}}'",
        f"def get_user_name_{task_id}(user_id: int) -> str:\n    \"\"\"Fetches user name by ID.\"\"\"\n    users = {{1: 'Alice', 2: 'Bob'}}\n    # Look up the user in the dictionary\n    name = users.get(user_id, 'Unknown')\n    return f'User: {{name}}'",
    ]
    return random.choice(tasks)

def generate_claude_style(task_id):
    """Claude: Concise, functional style, minimal comments."""
    tasks = [
        f"def scale_values_{task_id}(values, factor):\n    return [v * factor for v in values]",
        f"def filter_evens_{task_id}(numbers):\n    return list(filter(lambda x: x % 2 == 0, numbers))",
    ]
    return random.choice(tasks)

def generate_llama_style(task_id):
    """Llama: Standard, sometimes uses old-style formatting, mixed snake/camel."""
    tasks = [
        f"def CalculateArea_{task_id}(radius):\n    import math\n    area = math.pi * (radius ** 2)\n    return 'Area: %.2f' % area",
        f"def list_files_{task_id}(path):\n    import os\n    return os.listdir(path)",
    ]
    return random.choice(tasks)

def generate_deepseek_style(task_id):
    """DeepSeek: Highly optimized, uses bitwise ops or list comps, very technical."""
    tasks = [
        f"def is_power_of_two_{task_id}(n):\n    return n > 0 and (n & (n - 1)) == 0",
        f"def fast_fib_{task_id}(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    ]
    return random.choice(tasks)

def generate_gemini_style(task_id):
    """Gemini: Practical, clear, good documentation, balanced style."""
    tasks = [
        f"def find_max_{task_id}(numbers):\n    if not numbers: return None\n    curr_max = numbers[0]\n    for n in numbers:\n        if n > curr_max: curr_max = n\n    return curr_max",
        f"def greet_user_{task_id}(name):\n    return f'Hello, {{name}}! Welcome to the system.'",
    ]
    return random.choice(tasks)

# --- Real LLM Sourcing ---

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
    "Write a Python function to find the most frequent element in a list."
]

def fetch_real_samples(model_type, api_key):
    samples = []
    if not api_key:
        return samples

    print(f"Sourcing real samples for {model_type}...")
    
    try:
        if model_type == "gpt4o" and OpenAI:
            client = OpenAI(api_key=api_key)
            for prompt in PROMPTS:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": f"{prompt} Output ONLY the Python code, no explanation."}]
                )
                code = response.choices[0].message.content.strip()
                # Clean up markdown code blocks if present
                code = code.replace("```python", "").replace("```", "").strip()
                samples.append(code)
                time.sleep(1) # Rate limiting safety

        elif model_type == "claude35" and anthropic:
            client = anthropic.Anthropic(api_key=api_key)
            for prompt in PROMPTS:
                # Try Haiku as it is usually available
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": f"{prompt} Output ONLY the Python code, no explanation."}]
                )
                code = response.content[0].text.strip()
                code = code.replace("```python", "").replace("```", "").strip()
                samples.append(code)
                time.sleep(1)

        elif model_type == "gemini" and genai:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-pro')
            for prompt in PROMPTS:
                response = model.generate_content(f"{prompt} Output ONLY the Python code, no explanation.")
                code = response.text.strip()
                code = code.replace("```python", "").replace("```", "").strip()
                samples.append(code)
                time.sleep(1)

    except Exception as e:
        print(f"Error fetching real samples for {model_type}: {e}")
    
    return samples

# --- Main Execution ---

def main():
    base = Path(__file__).parent
    output_dir = base / "research/data/raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration: (name, generator, api_key_env)
    configs = [
        ("gpt4o", generate_gpt4o_style, "OPENAI_API_KEY"),
        ("claude35", generate_claude_style, "ANTHROPIC_API_KEY"),
        ("gemini", generate_gemini_style, "GEMINI_API_KEY"),
        ("llama3", generate_llama_style, None),
        ("deepseek", generate_deepseek_style, None)
    ]
    
    TOTAL_TARGET = 50
    
    for model_name, generator, env_var in configs:
        real_samples = []
        if env_var:
            api_key = os.environ.get(env_var)
            if api_key:
                real_samples = fetch_real_samples(model_name, api_key)
        
        real_count = len(real_samples)
        
        # Save real samples
        for i, code in enumerate(real_samples):
            filename = f"ai_{model_name}_{i}.py"
            with open(output_dir / filename, "w") as f:
                f.write(code)
        
        # Fill the rest with synthetic
        for i in range(real_count, TOTAL_TARGET):
            filename = f"ai_{model_name}_{i}.py"
            content = generator(i)
            with open(output_dir / filename, "w") as f:
                f.write(content)
        
        print(f"Generated {TOTAL_TARGET} samples for {model_name} ({real_count} real, {TOTAL_TARGET - real_count} synthetic)")

if __name__ == "__main__":
    main()
