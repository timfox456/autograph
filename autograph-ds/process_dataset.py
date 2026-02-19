import os
from pathlib import Path
import pandas as pd
import json
from src.features.extractor import LogicalDNAExtractor
from src.utils import flatten_dna
from collections import Counter

MIN_LINES = 5  # Skip samples below this threshold — too little signal for feature extraction


def main():
    extractor = LogicalDNAExtractor()
    base = Path(__file__).parent
    raw_dir = base / "research/data/raw"
    processed_dir = base / "research/data/processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. First pass: Collect trigrams and extract DNA
    all_dnas = []
    global_trigram_counts = Counter()
    skipped = Counter()

    files = [f for f in os.listdir(raw_dir) if f.endswith(".py")]
    print(f"Found {len(files)} raw files. Filtering with MIN_LINES={MIN_LINES}...")

    for filename in files:
        file_path = raw_dir / filename
        with open(file_path, "r") as f:
            code = f.read()

        line_count = len(code.splitlines())
        if line_count < MIN_LINES:
            # Derive identity for skip reporting
            parts = filename.replace(".py", "").split("_")
            identity_key = "_".join(parts[:2]) if len(parts) >= 2 else filename
            skipped[identity_key] += 1
            continue

        try:
            dna = extractor.extract(code)
            # Collect trigrams for filtering
            if "top_trigrams" in dna:
                global_trigram_counts.update(dna["top_trigrams"])

            dna["_filename"] = filename  # Temp storage
            all_dnas.append(dna)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    total_skipped = sum(skipped.values())
    print(f"Skipped {total_skipped} files below {MIN_LINES} lines:")
    for identity, count in sorted(skipped.items(), key=lambda x: -x[1]):
        print(f"  {identity}: {count} skipped")

    # 2. Select Top 500 Trigrams
    top_trigrams = [t for t, count in global_trigram_counts.most_common(500)]
    print(f"Selected Top 500 trigrams out of {len(global_trigram_counts)} unique sequences.")
    
    # 3. Flatten and filter
    rows = []
    for dna in all_dnas:
        filename = dna.pop("_filename")
        
        # Filter trigrams
        if "top_trigrams" in dna:
            raw_trigrams = dna.pop("top_trigrams")
            filtered_trigrams = {t: raw_trigrams.get(t, 0) for t in top_trigrams}
            dna["top_trigrams"] = filtered_trigrams
            
        flat_dna = flatten_dna(dna)
        
        # Labeling
        if filename.startswith("human_"):
            flat_dna["label"] = "human"
            parts = filename.split("_")
            if len(parts) >= 2:
                flat_dna["identity"] = parts[1]
            else:
                flat_dna["identity"] = filename.replace("human_", "").replace(".py", "")
        else:
            flat_dna["label"] = "ai"
            parts = filename.split("_")
            if len(parts) >= 2:
                flat_dna["identity"] = parts[1]
            else:
                flat_dna["identity"] = filename.replace("ai_", "").replace(".py", "")
        
        flat_dna["identity"] = flat_dna["identity"].replace(".py", "").replace("human_", "")
        flat_dna["filename"] = filename
        rows.append(flat_dna)
        
    df = pd.DataFrame(rows)
    df = df.fillna(0)
    
    output_path = processed_dir / "dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Processed {len(rows)} files into {output_path}")

if __name__ == "__main__":
    main()
