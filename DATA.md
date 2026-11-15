# DATA.md — data handling rules

## Nothing data-like is committed to git

Datasets, rendered shards, real document images, training manifests, pools, checkpoints and run
outputs never enter the repository. The one committed manifest is the image-free evaluation list
[`benchmarks/frozen-test-v2/manifest.csv`](benchmarks/frozen-test-v2/manifest.csv), which stores
only public `Keylab/OCSR-Benchmarks` row IDs and SMILES (no images). This rule is enforced twice:

1. `.gitignore` excludes `/data/`, `/data_cache/`, `/real_data/`, `/real/`, `shards/`, `runs/`,
   `checkpoints/`, `*.tar`, `*.idx.json`, `*.tsv.gz`, `*.ckpt`, `*.pt`.
2. A pre-commit guard, `scripts/check_no_data.py`, refuses staged files that live in those
   directories, have archive/checkpoint suffixes, are raster images outside `benchmarks/` (or
   inside it with "real" in the path), or exceed 5 MB. Install it once per clone:

   ```bash
   bash scripts/install_git_hooks.sh      # or: pre-commit install (uses .pre-commit-config.yaml)
   ```

Only our own synthetic figures and aggregate plots may be committed under `benchmarks/`.

## Real journal / patent figures

Real depictions (e.g. the ACS, CLEF-IP and USPTO sets used by arXiv:2608.09100, hand-drawn
collections) may carry copyright or redistribution restrictions from publishers or patent
offices. Keep them on private storage (local disk, the Colab runtime, a private Drive folder),
do not upload them to public places, and do not include individual real images in figures that
are committed or published unless the licence allows it.

## Synthetic data provenance

| source | licence / origin | used for |
|--------|------------------|----------|
| ZINC250k (`aspuru-guzik-group/chemical_vae`) | ZINC-derived, redistributed on GitHub | molecule pool |
| MOSES `dataset_v1.csv` (`molecularsets/moses`) | ZINC clean leads, MIT-licensed repo | molecule pool |

Images are rendered locally with RDKit; see `ouroboros/data/render.py` for the style ranges.

## Real-data manifest format

See `ouroboros/data/real.py` (module docstring). Minimal CSV:

```csv
image,smiles,source,split,id
images/0001.png,CC(=O)Oc1ccccc1C(=O)O,USPTO,test,US1234567-C1
```

Ingest with `python scripts/ingest_real.py --manifest M.csv --out real/<name>`; rows that fail are
logged to `real/<name>/ingest_failures.jsonl` with a reason and counted in `ingest_stats.json`.

### Real corpus on the Hugging Face Hub

The user's real corpus is the HF dataset `yaniguan/ocsr-dataset` (made by the user). It is
downloaded only inside Colab (`notebooks/04_real_data.ipynb`; private access via a Colab secret
`HF_TOKEN`), never into the repository. `scripts/hf_to_manifest.py inspect` reports the detected
image / SMILES / source / split columns; `convert` writes one manifest per source
(`/content/hf_real/manifests/<source>.csv`), and each is ingested to
`MyDrive/ouroboros/real/<source>`. Rows with no recognisable split are assigned `test`, so an
unlabelled row is never trained on. Snapshot, extracted images and manifests stay on the Colab
disk / Drive; `hf/`, `hf_real/`, `*.parquet`, `*.arrow` are git-ignored and refused by the
pre-commit guard.

### VERDICT-PMC real training data (Oct 30)

`colab/finetune_real_pmc.ipynb` downloads the public parquet of `yaniguan/ocsr-dataset` at a pinned
commit, runs `scripts/prepare_pmc.py` (removes every row whose molecule's connectivity block matches a
held-out test molecule, from train AND val; drops overlong labels; makes val molecule-disjoint from
train) and ingests it with `scripts/ingest_real.py`. Train rows are used for training, val rows only for
checkpoint selection. Images, parquet and shards stay on the Colab disk / Drive.

## Real evaluation sets (`dev-test` vs `frozen-test-v2`)

Two seeded samples from `Keylab/OCSR-Benchmarks` (`USPTO`, `UOB`, `ACS`) are used for real-figure
evaluation:

- **`subset100` (`dev-test`, 298 valid images: ACS 100, UOB 100, USPTO 98):** `seed=0` sample with
  pinned manifest SHA-256 `957f0042825dab214aad2463de2974d7b4f23a1fcbf5d52ab78be2526cc3fc61`
  (`EXPECTED_DEV_TEST_MANIFEST_SHA256` in `ouroboros/data/real.py`). Because many rounds of model
  and decoding choices were compared on this set, keep it as a `dev-test` set for quick checks and
  historical comparison, not as a clean held-out set.
- **`frozen-test-v2` (held-out release test, 300 valid images: ACS 100, UOB 100, USPTO 100):**
  `seed=2026` sample in
  [`benchmarks/frozen-test-v2/manifest.csv`](benchmarks/frozen-test-v2/manifest.csv), pinned as
  `EXPECTED_FROZEN_TEST_V2_MANIFEST_SHA256`
  (`e4a24c68b3d893498b0df727bdb5b77eed0b98e66f6898f21d17aecf01ee5e1a`). It has zero row, full
  InChIKey, or 14-character connectivity-block overlap with `subset100`, `yaniguan/ocsr-dataset`
  (`train` and `val`), and `zinc250k` (see
  [`benchmarks/frozen-test-v2/leakage.json`](benchmarks/frozen-test-v2/leakage.json)).

**Score-once rule:** keep `frozen-test-v2` untouched during development. Score it **at most once per
release**, only after all training, checkpoint, and decoding choices are fixed on validation.
Materialize its images locally on demand (never into git) with:

```bash
python scripts/fetch_real_subset.py \
    --from-manifest benchmarks/frozen-test-v2/manifest.csv \
    --out real_data/frozen_test_v2
python scripts/ingest_real.py \
    --manifest real_data/frozen_test_v2/manifest.csv \
    --out real/frozen_test_v2
```

## Leakage control

`scripts/check_leakage.py` compares standard InChIKeys of every real-document eval set with every
training set (synthetic and real), prints the overlap table and exits nonzero on any overlap
(`--fail-on-connectivity` also rejects stereoisomer / connectivity-block matches).
`--dump-eval-keys` writes the eval keys (include both `subset100` and
`benchmarks/frozen-test-v2/manifest.csv`), which `scripts/build_dataset.py --exclude-keys` and
`scripts/prepare_pmc.py --exclude-keys` remove by connectivity block before building splits.
Shards rendered earlier can instead be filtered at load time
(`ShardDataset(..., exclude_key14=...)`).

## Model weights

MACE-OFF23 weights (downloaded by `mace-torch` from `ACEsuit/mace-off` on first use) are
distributed under the Academic Software License (ASL, https://github.com/gabor1/ASL): academic,
non-commercial use. They are cached in `~/.cache/mace/` and never committed.

Download identities (pinned in `ouroboros/data/sources.py`): ZINC250k revision
`37b9f96470d4471c0593cffefa448e0a8a184ef6`, Git blob
`d31726bf929d35cb52fa109db90411c90a000dfe` (22,606,589 bytes); MOSES revision
`a3866ff959325b60e36f8b15beebdaa3bfaea188`, Git LFS SHA256
`bb47a94d347afd476d3828b5e26dceeabc42a2d8cf92a791d00349f22fea0d8b` (84,482,588 bytes).
These identities were read from the upstream GitHub content metadata. Resumable raw downloads,
filtering SQLite caches and checksum receipts remain in private data storage; see
[reliable runs](docs/reliability.md).
