from src.engine import AttestationEngine
import json

def run_demo():
    engine = AttestationEngine(
        matcher_path="autograph-ds/research/models/matcher.joblib",
        consistency_dir="autograph-ds/research/models"
    )

    print("=== Autograph Attribution Model Demo ===\n")

    # Scenario 1: Verified Match (Mariusz)
    print("Scenario 1: Verified Match (Mariusz code claimed as mariusz)")
    with open("autograph-ds/research/data/raw/human_mariusz_0.py", "r") as f:
        mariusz_code = f.read()
    
    report1 = engine.attest(mariusz_code, "mariusz")
    print(json.dumps(report1, indent=2))
    print("-" * 40)

    # Scenario 2: Mismatch (GPT-4 claimed as Mariusz)
    print("Scenario 2: Mismatch (GPT-4 code claimed as mariusz)")
    with open("autograph-ds/research/data/raw/ai_gpt4o.py", "r") as f:
        gpt_code = f.read()
    
    report2 = engine.attest(gpt_code, "mariusz")
    print(json.dumps(report2, indent=2))
    print("-" * 40)


    # Scenario 2: Mismatch (GPT-4 claimed as Django)
    print("Scenario 2: Mismatch (GPT-4 code claimed as Django)")
    with open("autograph-ds/research/data/raw/ai_gpt4o.py", "r") as f:
        gpt_code = f.read()
    
    report2 = engine.attest(gpt_code, "django")
    print(json.dumps(report2, indent=2))
    print("-" * 40)

    # Scenario 3: Spoofing (DeepSeek claimed as GPT-4)
    print("Scenario 3: Spoofing (DeepSeek code claimed as gpt4o)")
    with open("autograph-ds/research/data/raw/ai_deepseek_v3.py", "r") as f:
        # Manually add a DeepSeek marker to ensure spoofing detection triggers
        ds_code = f.read() + "\n# -*- coding: utf-8 -*-"
    
    report3 = engine.attest(ds_code, "gpt4o")
    print(json.dumps(report3, indent=2))
    print("-" * 40)

    # Scenario 4: Privacy Sparsity (Requests with only Structural features)
    print("Scenario 4: Privacy Sparsity (Requests code claimed as Requests, Structural ONLY)")
    with open("autograph-ds/research/data/raw/human_requests.py", "r") as f:
        req_code = f.read()
    
    report4 = engine.attest(req_code, "requests", enabled_buckets=["structural_topology"])
    print(json.dumps(report4, indent=2))
    print("-" * 40)

if __name__ == "__main__":
    run_demo()
