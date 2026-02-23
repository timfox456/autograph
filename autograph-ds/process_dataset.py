import os
from pathlib import Path
import pandas as pd
import json
from src.features.extractor import LogicalDNAExtractor
from src.utils import flatten_dna
from collections import Counter

# Global caps to prevent dimensionality explosion from dynamic keys
TOP_TRIGRAMS_K = 150
TOP_API_CALLS_K = 80          # limit semantic call_freq_* features
TOP_EXCEPTION_TYPES_K = 20    # limit exception_type_*_count features
TOP_NODE_TYPES_K = 120        # limit node_* (AST node types)

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
    global_call_counts = Counter()
    global_exception_counts = Counter()
    global_node_type_counts = Counter()
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

            # Collect high-cardinality semantic keys for global top-K filtering
            # 1) API calls: keys like 'call_freq_<safe_name>' with float ratios
            for k, v in dna.items():
                if isinstance(k, str) and k.startswith("call_freq_") and not k.startswith("call_freq_placeholder_"):
                    # Accumulate weighted by value; counts are normalized per-file,
                    # but summing across files yields a reasonable global importance
                    try:
                        global_call_counts[k] += float(v)
                    except Exception:
                        pass

                # 2) Exception types: keys like 'exception_type_<Name>_count'
                if isinstance(k, str) and k.startswith("exception_type_") and k.endswith("_count"):
                    try:
                        global_exception_counts[k] += int(v)
                    except Exception:
                        # Some implementations may store floats; fallback to float accumulation
                        try:
                            global_exception_counts[k] += float(v)
                        except Exception:
                            pass

            # 3) AST node types live under node_type_counts dict
            if isinstance(dna.get("node_type_counts"), dict):
                for node_type, count in dna["node_type_counts"].items():
                    try:
                        global_node_type_counts[node_type] += float(count)
                    except Exception:
                        pass

            dna["_filename"] = filename  # Temp storage
            all_dnas.append(dna)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    total_skipped = sum(skipped.values())
    print(f"Skipped {total_skipped} files below {MIN_LINES} lines:")
    for identity, count in sorted(skipped.items(), key=lambda x: -x[1]):
        print(f"  {identity}: {count} skipped")

    # 2. Select Top-K for trigrams and other dynamic features
    top_trigrams = [t for t, count in global_trigram_counts.most_common(TOP_TRIGRAMS_K)]
    print(f"Selected Top {TOP_TRIGRAMS_K} trigrams out of {len(global_trigram_counts)} unique sequences.")

    top_calls = [k for k, _ in global_call_counts.most_common(TOP_API_CALLS_K)]
    print(f"Selected Top {TOP_API_CALLS_K} API call features out of {len(global_call_counts)} candidates.")

    top_exceptions = [k for k, _ in global_exception_counts.most_common(TOP_EXCEPTION_TYPES_K)]
    print(f"Selected Top {TOP_EXCEPTION_TYPES_K} exception type features out of {len(global_exception_counts)} candidates.")

    top_node_types = [k for k, _ in global_node_type_counts.most_common(TOP_NODE_TYPES_K)]
    print(f"Selected Top {TOP_NODE_TYPES_K} AST node types out of {len(global_node_type_counts)} seen.")
    
    # 3. Flatten and filter
    rows = []
    for dna in all_dnas:
        filename = dna.pop("_filename")
        
        # Filter trigrams
        if "top_trigrams" in dna:
            raw_trigrams = dna.pop("top_trigrams")
            filtered_trigrams = {t: raw_trigrams.get(t, 0) for t in top_trigrams}
            dna["top_trigrams"] = filtered_trigrams

        # Filter API call features to top-K
        for k in list(dna.keys()):
            if isinstance(k, str) and k.startswith("call_freq_"):
                if k.startswith("call_freq_placeholder_"):
                    # Always drop placeholder features
                    dna.pop(k, None)
                elif k not in top_calls:
                    dna.pop(k, None)

        # Filter exception type features to top-K
        for k in list(dna.keys()):
            if isinstance(k, str) and k.startswith("exception_type_") and k.endswith("_count"):
                if k not in top_exceptions:
                    dna.pop(k, None)

        # Filter AST node types to top-K before flattening
        if isinstance(dna.get("node_type_counts"), dict):
            node_counts = dna["node_type_counts"]
            dna["node_type_counts"] = {nt: node_counts.get(nt, 0) for nt in top_node_types}

        flat_dna = flatten_dna(dna)
        
        # Labeling
        # Parse identity from filename: human_{identity}_{index}.py or ai_{identity}_{index}.py
        # Handles multi-part identities like "deepseek_v3" correctly
        if filename.startswith("human_"):
            flat_dna["label"] = "human"
            # Remove prefix and extension, extract identity (everything before _{number}.py)
            name_without_prefix = filename[6:]  # Remove "human_"
        else:
            flat_dna["label"] = "ai"
            name_without_prefix = filename[3:]  # Remove "ai_"
        
        # Remove .py extension
        name_without_ext = name_without_prefix.replace(".py", "")
        # Split by _ and reconstruct identity (all parts except the last number)
        parts = name_without_ext.split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            # Last part is the index number, identity is everything before it
            flat_dna["identity"] = "_".join(parts[:-1])
        elif len(parts) >= 1:
            # Fallback: just take everything (no index)
            flat_dna["identity"] = "_".join(parts)
        else:
            flat_dna["identity"] = name_without_ext
        flat_dna["filename"] = filename
        rows.append(flat_dna)
        
    df = pd.DataFrame(rows)
    df = df.fillna(0)
    
    output_path = processed_dir / "dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Processed {len(rows)} files into {output_path}")

if __name__ == "__main__":
    main()
