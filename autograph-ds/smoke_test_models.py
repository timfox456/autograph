#!/usr/bin/env python3
"""Quick smoke test for model-based embedders.

Checks device detection and verifies that each requested embedder can
produce an embedding of expected shape for a tiny code snippet.

Usage:
  python smoke_test_models.py --models codebert graphcodebert unixcoder clave

Exits non-zero if any requested model fails the smoke test.
Also writes a brief markdown report under research/model_analysis/reports/.
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np

# Proactive dependency check: provide a clear message if Torch isn't installed
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    print(
        "Missing dependency: 'torch' is required for model smoke tests.\n"
        "Install model dependencies first, e.g.:\n\n"
        "  - Using uv:    uv pip install -r requirements-model.txt\n"
        "  - Using pip:   pip install -r requirements-model.txt\n\n"
        "If you prefer a CPU-only Torch wheel:\n"
        "  pip install --index-url https://download.pytorch.org/whl/cpu torch\n",
        file=sys.stderr,
    )
    raise SystemExit(2)

sys.path.insert(0, str(Path(__file__).parent))

from src.model_analysis.device import get_device, get_device_info
from src.model_analysis.embedders import get_embedder
from src.model_analysis.clave_embedder import get_clave_embedder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("smoke_test")

REPORTS_DIR = Path("research/model_analysis/reports")


def run_smoke(models):
    device = get_device()
    info = get_device_info(device)
    logger.info(f"Device: {info['name']} ({info['type']})")

    sample_code = """
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a
""".strip()

    passed = {}
    details = []

    for model in models:
        try:
            if model == "clave":
                emb = get_clave_embedder()
                if not emb.is_available():
                    passed[model] = False
                    details.append(f"{model}: not available (download or deps missing)")
                    continue
            else:
                emb = get_embedder(model, device)

            vec = emb.embed(sample_code)
            ok = isinstance(vec, np.ndarray) and vec.ndim == 1 and vec.size > 0 and np.isfinite(vec).all()
            passed[model] = bool(ok)
            details.append(f"{model}: shape={vec.shape if isinstance(vec, np.ndarray) else 'N/A'} ok={ok}")
        except Exception as e:
            passed[model] = False
            details.append(f"{model}: error={e}")

    return passed, details, info


def write_report(passed, details, info):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"smoke_test_{stamp}.md"

    lines = []
    lines.append(f"# Model Smoke Test — {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"- Device: {info['name']} ({info['type']})")
    lines.append("")
    lines.append("## Results")
    for k, v in passed.items():
        lines.append(f"- {k}: {'✅' if v else '❌'}")
    lines.append("")
    lines.append("## Details")
    for d in details:
        lines.append(f"- {d}")

    out.write_text("\n".join(lines))
    logger.info(f"Smoke test report written to {out}")


def main():
    parser = argparse.ArgumentParser(description="Smoke test model embedders")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["codebert", "graphcodebert", "unixcoder", "clave", "all"],
        default=["all"],
    )
    args = parser.parse_args()
    models = ["codebert", "graphcodebert", "unixcoder", "clave"] if "all" in args.models else args.models

    passed, details, info = run_smoke(models)
    write_report(passed, details, info)

    ok = all(passed.values()) if passed else False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
