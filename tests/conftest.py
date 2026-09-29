"""Shared fixtures: a tiny synthetic dataset built with the real pipeline."""

import random

import pytest

from ouroboros.data import build


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("ds")
    rng = random.Random(1)
    smiles = []
    frags = ["C", "CC", "O", "N", "Cl", "F", "C(C)O", "C=CC", "c1ccccc1", "C1CCCC1", "C(=O)N"]
    while len(smiles) < 600:
        smiles.append("CC" + "".join(rng.choice(frags) for _ in range(rng.randint(2, 6))))
    build.prepare_pool([("synthetic", smiles)], root / "pool.tsv.gz", workers=2)
    pool = build.read_pool(root / "pool.tsv.gz")
    cfg = build.ComposeConfig(
        stereo_fraction=0.4,
        sizes={"train": 100, "val": 20, "test": 20},
        val_frac=0.1,
        test_frac=0.1,
    )
    stats = build.compose(pool, cfg, root / "manifests")
    for split in build.SPLITS:
        rows = build.read_manifest(root / "manifests" / f"{split}.tsv")
        build.render_shards(rows, split, root / "shards", shard_size=40, size=64, workers=2)
    return root, stats
