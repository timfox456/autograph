import os
import pandas as pd
import json
from src.features.extractor import LogicalDNAExtractor
from src.utils import flatten_dna

def main():
    extractor = LogicalDNAExtractor()
    raw_dir = "autograph-ds/research/data/raw"
    processed_dir = "autograph-ds/research/data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    rows = []
    for filename in os.listdir(raw_dir):
        if not filename.endswith(".py"):
            continue
            
        file_path = os.path.join(raw_dir, filename)
        with open(file_path, 'r') as f:
            code = f.read()
            
        dna = extractor.extract(code)
        flat_dna = flatten_dna(dna)
        
        # Labeling
        if filename.startswith("human_"):
            flat_dna["label"] = "human"
            # Group by author name: human_mariusz_0.py -> mariusz
            parts = filename.split("_")
            if len(parts) >= 2:
                flat_dna["identity"] = parts[1]
            else:
                flat_dna["identity"] = filename.replace("human_", "").replace(".py", "")
        else:
            flat_dna["label"] = "ai"
            flat_dna["identity"] = filename.replace("ai_", "").replace(".py", "")
        
        # Final cleanup for identity names
        flat_dna["identity"] = flat_dna["identity"].replace(".py", "").replace("human_", "")
            
        flat_dna["filename"] = filename
        rows.append(flat_dna)
        
    df = pd.DataFrame(rows)
    # Fill NaNs with 0 for counts/distributions
    df = df.fillna(0)
    
    output_path = os.path.join(processed_dir, "dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"Processed {len(rows)} files into {output_path}")

if __name__ == "__main__":
    main()
