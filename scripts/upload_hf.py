"""Upload an export folder (scripts/export_hf_release.py) to the Hugging Face Hub.

    python scripts/upload_hf.py dist/hf/C                       # DRY RUN: list files and target
    python scripts/upload_hf.py dist/hf/C --private --yes       # create the repo and upload

Dry run is the default: it verifies ``checksums.txt``, prints every file that would be uploaded
and the target repository, and neither imports ``huggingface_hub`` nor touches the network. Only
``--yes`` uploads. Only the files listed in ``checksums.txt`` (plus that file) are sent, so stray
files in the folder never leave the machine.

Authentication is left entirely to huggingface_hub: the ``HF_TOKEN`` environment variable or a
prior ``huggingface-cli login``. This script never reads, accepts or prints a token.
``--private`` applies when the repository is created; an existing repository keeps its visibility.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ouroboros.hf import CHECKSUMS_FILE, CONFIG_FILE, read_checksums, verify_checksums

DEFAULT_REPO_ID = "yaniguan/ouroboros-ocsr-encoder"
REQUIRED = (CONFIG_FILE, "vocab.json", "README.md")


def plan(folder: Path) -> list[str]:
    """Files to upload (verified against checksums.txt), checksums.txt last."""
    if not (folder / CHECKSUMS_FILE).is_file():
        raise SystemExit(f"{folder} has no {CHECKSUMS_FILE}: run scripts/export_hf_release.py")
    names = sorted(read_checksums(folder))
    missing = [n for n in REQUIRED if n not in names]
    if missing or not any(n.startswith("model.") or n == "pytorch_model.bin" for n in names):
        raise SystemExit(f"{folder} is not a complete export (missing {missing or 'weights'})")
    try:
        verify_checksums(folder, names)
    except ValueError as e:
        raise SystemExit(f"refusing to upload: {e}") from e
    return [*names, CHECKSUMS_FILE]


def upload(folder: Path, files: list[str], repo_id: str, private: bool, message: str) -> str:
    from huggingface_hub import HfApi  # imported only for a real upload

    api = HfApi()  # token: HF_TOKEN or the huggingface-cli login cache, resolved by the library
    url = api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=files,
        commit_message=message,
    )
    return str(getattr(commit, "commit_url", None) or url)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("folder", type=Path, help="export folder")
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--private", action="store_true", help="create the repository as private")
    ap.add_argument("--message", default=None, help="commit message")
    ap.add_argument("--yes", action="store_true", help="really upload (default: dry run)")
    a = ap.parse_args(argv)
    folder = a.folder
    files = plan(folder)
    config = json.loads((folder / CONFIG_FILE).read_text())
    message = a.message or (
        f"Upload Ouroboros arm {config.get('arm')} ({config.get('encoder_type')}), "
        f"step {config.get('step')}"
    )
    print(f"target: https://huggingface.co/{a.repo_id} "
          f"({'private' if a.private else 'public'} if created now)")
    print(f"commit message: {message}")
    for name in files:
        print(f"  {(folder / name).stat().st_size:>14,d}  {name}")
    if not a.yes:
        print("DRY RUN: nothing uploaded. Re-run with --yes to upload.")
        return 0
    print(f"uploaded: {upload(folder, files, a.repo_id, a.private, message)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
