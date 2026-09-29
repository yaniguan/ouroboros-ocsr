"""Hugging Face dataset -> real-data manifests for ``scripts/ingest_real.py`` (Am1-A, U1).

    # 1) look at what the dataset contains (no files written)
    python scripts/hf_to_manifest.py inspect --repo yaniguan/ocsr-dataset --local /content/hf/ocsr
    # 2) extract images + write one manifest per source
    python scripts/hf_to_manifest.py convert --local /content/hf/ocsr --out /content/hf_real
    # 3) per source X:
    python scripts/ingest_real.py --manifest /content/hf_real/manifests/X.csv --out real/X

``--repo`` downloads a snapshot first (``huggingface_hub.snapshot_download``; a private dataset
needs ``HF_TOKEN`` in the environment). Column / split / source detection is automatic and can be
overridden (``--image-col``, ``--smiles-col``, ``--source-col``, ``--split-col``, ``--id-col``,
``--split-map validation=test``). Nothing written here may be committed (DATA.md).
"""

from __future__ import annotations

import argparse
import json
import os

from ouroboros.data.hf_manifest import inspect_snapshot, write_manifests


def _download(repo: str, local: str, revision: str | None) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo,
        repo_type="dataset",
        local_dir=local,
        revision=revision,
        token=os.environ.get("HF_TOKEN") or None,
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", choices=["inspect", "convert"])
    ap.add_argument("--local", required=True, help="local snapshot directory")
    ap.add_argument("--repo", help="HF dataset id to download into --local first")
    ap.add_argument("--revision")
    ap.add_argument("--out", help="convert: output directory (images/ + manifests/)")
    for role in ("image", "smiles", "source", "split", "id"):
        ap.add_argument(f"--{role}-col")
    ap.add_argument("--default-source", default="real")
    ap.add_argument("--default-split", default="test")
    ap.add_argument("--split-map", nargs="*", default=[], help="e.g. val=test train=drop")
    a = ap.parse_args(argv)
    if a.repo:
        _download(a.repo, a.local, a.revision)
    overrides = {r: getattr(a, f"{r}_col") for r in ("image", "smiles", "source", "split", "id")}
    if a.mode == "inspect":
        print(json.dumps(inspect_snapshot(a.local, overrides=overrides), indent=1, default=str))
        return
    if not a.out:
        ap.error("convert needs --out")
    split_map = dict(kv.split("=", 1) for kv in a.split_map)
    stats = write_manifests(
        a.local,
        a.out,
        overrides=overrides,
        default_source=a.default_source,
        default_split=a.default_split,
        split_map=split_map,
    )
    print(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
