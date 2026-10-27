"""Explicit external source folders and fresh outputs for manuscript renderers."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(description, inputs):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    for name in inputs:
        if not (args.data_dir / name).is_file():
            parser.error(f"Missing input: {name}")
    if args.outdir.exists():
        parser.error("Use a new output directory")
    args.outdir.mkdir(parents=True)
    return args.data_dir, args.outdir


def finish(data, out, inputs, script):
    receipt = {
        "schema": "manuscript-figure-render/1",
        "inputs_sha256": {name: sha(data / name) for name in inputs},
        "code_sha256": {script.name: sha(script), "_figure_io.py": sha(Path(__file__))},
        "outputs_sha256": {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
        "model_calls": 0,
        "analysis_refitted": False,
        "human_labels_changed": False,
        "scope": "Rendering supplied saved summaries; not an end-to-end analysis validation.",
    }
    (out / "render_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(out)
