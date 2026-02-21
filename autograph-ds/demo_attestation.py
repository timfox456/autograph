from src.engine import AttestationEngine
from pathlib import Path
import json

def print_attestation_report(report, scenario_title):
    """Pretty-print attestation report with key fields highlighted."""
    print(f"\n{scenario_title}")
    print("=" * 60)
    
    # Core verdict and confidence
    print(f"Verdict: {report['verdict']}")
    print(f"Claimed Identity: {report['claimed_identity']}")
    print(f"Detected Identity: {report['detected_identity']}")
    print(f"Match: {report['is_match']}")
    
    # Confidence metrics
    print(f"\nConfidence Metrics:")
    print(f"  - Confidence: {report['confidence']:.2%}")
    print(f"  - Corpus Probability: {report['corpus_probability']:.2%}")
    print(f"  - Corpus Percentile: {report['corpus_percentile']:.2%}")
    
    # AI Detection
    print(f"\nAI Detection:")
    print(f"  - AI Probability: {report['ai_probability']:.2%}")
    print(f"  - Detected AI Identity: {report['detected_ai_identity']}")
    if report['detected_ai_identity']:
        print(f"  - Detected AI Probability: {report['detected_ai_probability']:.2%}")
    
    # Consistency and flags
    print(f"\nConsistency Check: {report['consistency']}")
    if report['flags']:
        print(f"Flags: {report['flags']}")
    if report['markers']:
        print(f"Markers: {report['markers']}")
    
    print("-" * 60)

def run_demo():
    base = Path(__file__).parent
    models_dir = base / "research/models"
    engine = AttestationEngine(
        matcher_path=str(models_dir / "matcher.joblib"),
        consistency_dir=str(models_dir)
    )

    print("\n" + "=" * 60)
    print("=== Autograph Attribution Model Demo ===")
    print("Showcasing new fields: corpus_probability, ai_probability, detected_ai_identity")
    print("=" * 60)

    # Scenario 1: Verified Match (Mariusz)
    print("\nScenario 1: Verified Match (Mariusz code claimed as mariusz)")
    with open(base / "research/data/raw/human_mariusz_0.py", "r") as f:
        mariusz_code = f.read()
    
    report1 = engine.attest(mariusz_code, "mariusz")
    print_attestation_report(report1, "Scenario 1: Human Code - Verified Match")
    print("Full Report:")
    print(json.dumps(report1, indent=2))

    # Scenario 2: Mismatch (GPT-4 claimed as Mariusz)
    print("\nScenario 2: Mismatch (GPT-4 code claimed as mariusz)")
    with open(base / "research/data/raw/ai_gpt4o_0.py", "r") as f:
        gpt_code = f.read()
    
    report2 = engine.attest(gpt_code, "mariusz")
    print_attestation_report(report2, "Scenario 2: AI Code - Claimed as Human")
    print("Full Report:")
    print(json.dumps(report2, indent=2))

    # Scenario 3: Mismatch (GPT-4 claimed as Django)
    print("\nScenario 3: Mismatch (GPT-4 code claimed as Django)")
    with open(base / "research/data/raw/ai_gpt4o_1.py", "r") as f:
        gpt_code = f.read()
    
    report3 = engine.attest(gpt_code, "armin")
    print_attestation_report(report3, "Scenario 3: AI Code - Claimed as Different Human")
    print("Full Report:")
    print(json.dumps(report3, indent=2))

    # Scenario 4: Spoofing (Claude claimed as GPT-4)
    print("\nScenario 4: Spoofing (Claude code claimed as gpt4o)")
    with open(base / "research/data/raw/ai_claude_0.py", "r") as f:
        claude_code = f.read()
    
    report4 = engine.attest(claude_code, "gpt4o")
    print_attestation_report(report4, "Scenario 4: AI Spoofing - Claude Claimed as GPT-4")
    print("Full Report:")
    print(json.dumps(report4, indent=2))

    # Scenario 5: Privacy Sparsity (Hynek code with only Structural features)
    print("\nScenario 5: Privacy Sparsity (Hynek code claimed as Hynek, Structural ONLY)")
    with open(base / "research/data/raw/human_hynek_0.py", "r") as f:
        hynek_code = f.read()
    
    report5 = engine.attest(hynek_code, "hynek", enabled_buckets=["structural_topology"])
    print_attestation_report(report5, "Scenario 5: Privacy-Preserving Attestation")
    print("Full Report:")
    print(json.dumps(report5, indent=2))

    # Scenario 6: Human code with high corpus_probability
    print("\nScenario 6: Human Code with High Corpus Probability")
    print("(Using Armin code claimed as Armin - demonstrates strong corpus match)")
    with open(base / "research/data/raw/human_armin_0.py", "r") as f:
        human_code = f.read()
    
    report6 = engine.attest(human_code, "armin")
    print_attestation_report(report6, "Scenario 6: Strong Human Attribution")
    print("Key Insight: High corpus_probability indicates code strongly matches claimed identity's corpus")
    print("Full Report:")
    print(json.dumps(report6, indent=2))

    # Scenario 7: AI code showing high ai_probability
    print("\nScenario 7: AI Code with High AI Probability")
    print("(Using GPT-4 code claimed as GPT-4 - demonstrates AI detection)")
    with open(base / "research/data/raw/ai_gpt4o_2.py", "r") as f:
        ai_code = f.read()
    
    report7 = engine.attest(ai_code, "gpt4o")
    print_attestation_report(report7, "Scenario 7: Strong AI Detection")
    print("Key Insight: High ai_probability indicates code matches AI model patterns")
    print("Full Report:")
    print(json.dumps(report7, indent=2))

if __name__ == "__main__":
    run_demo()
