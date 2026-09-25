import math
import random

import numpy as np
import pytest
import torch
from rdkit import Chem

from ouroboros.data import build
from ouroboros.data.filter import filter_smiles
from ouroboros.data.loader import ResumableSampler, ShardDataset, collate, rotate_images
from ouroboros.data.render import RenderStyle, render, sample_style
from ouroboros.data.stereo import assign_random_stereo, defined_stereo, has_stereo, potential_stereo

# ----------------------------------------------------------------------------- filter


@pytest.mark.parametrize(
    "smi,ok,reason",
    [
        ("CCOC(=O)c1ccccc1", True, "ok"),
        ("CCCC", False, "too_small"),  # 4 heavy atoms
        ("CCCCC", True, "ok"),  # 5 heavy atoms: lower bound inclusive
        ("C" * 60, True, "ok"),  # upper bound inclusive
        ("C" * 61, False, "too_large"),
        ("CC[Si](C)(C)C", False, "element"),
        ("CCB(O)O", False, "element"),
        ("CC[Se]CC", False, "element"),
        ("C[CH]CCCC", False, "radical"),  # CH with two neighbours: one radical e-
        ("C[N+](C)(C)CCCC", False, "charged"),  # quaternary ammonium cannot be neutralized
        ("[13CH3]CCCCO", True, "ok"),  # isotopes are stripped by standardization
    ],
)
def test_filter_rules(smi, ok, reason):
    mol, why = filter_smiles(smi)
    assert (mol is not None) == ok
    assert why == reason


def test_filter_neutralizes_and_strips_salts():
    mol, why = filter_smiles("CCCC[NH3+].[Cl-]")
    assert why == "ok" and Chem.MolToSmiles(mol) == "CCCCN"
    mol, why = filter_smiles("CCCCC(=O)[O-].[Na+]")
    assert why == "ok" and Chem.MolToSmiles(mol) == "CCCCC(=O)O"
    mol, why = filter_smiles("CCc1ccc([N+](=O)[O-])cc1")  # nitro: charge-separated but neutral
    assert why == "ok"


# ----------------------------------------------------------------------------- stereo


def test_potential_and_assigned_stereo():
    mol = Chem.MolFromSmiles("CC=CC(Cl)C(O)c1ccccc1")
    assert potential_stereo(mol) == (2, 1)
    full = assign_random_stereo(mol, random.Random(0), tetrahedral=True)
    assert defined_stereo(full) == (2, 1)
    ez_only = assign_random_stereo(mol, random.Random(0), tetrahedral=False)
    assert defined_stereo(ez_only) == (0, 1)
    assert not has_stereo("c1ccccc1CC")


# ----------------------------------------------------------------------------- render

PLAIN = RenderStyle()  # no noise / blur / JPEG


def _enantiomer(smi: str) -> str:
    inv = smi.replace("@@", "\0").replace("@", "@@").replace("\0", "@")
    return Chem.CanonSmiles(inv)


@pytest.mark.parametrize(
    "smi", ["C[C@H](N)C(=O)O", "C/C=C/[C@@H](Cl)[C@H](O)c1ccccc1", "C[C@H]1CC[C@@H](C)CC1O"]
)
def test_label_and_mirror(smi):
    mol = Chem.MolFromSmiles(smi)
    r = render(mol, PLAIN, size=128)
    assert r.label == Chem.CanonSmiles(smi)
    rm = render(mol, PLAIN, size=128, mirror=True)
    assert rm.label == _enantiomer(smi)


def test_mirror_keeps_ez_and_achiral():
    for smi in ["C/C=C/CCCO", "CC(C)Cc1ccccc1"]:
        assert render(Chem.MolFromSmiles(smi), PLAIN, mirror=True).label == Chem.CanonSmiles(smi)


def test_render_inside_inscribed_circle_and_random_styles():
    rng = random.Random(0)
    mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)NCc1ccccc1")  # long, wide
    img = np.asarray(render(mol, PLAIN, size=256).image)
    yy, xx = np.mgrid[:256, :256]
    outside = (xx + 0.5 - 128) ** 2 + (yy + 0.5 - 128) ** 2 > 128**2
    assert img.shape == (256, 256)
    assert (img[outside] == 255).all() and (img < 128).any()
    for i in range(10):
        st = sample_style(rng)
        assert render(mol, st, size=96, seed=i).label == Chem.MolToSmiles(mol)


# ----------------------------------------------------------------------------- build


@pytest.fixture(scope="module")
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


def test_compose_fraction_prefix_and_disjoint_splits(tiny_dataset):
    root, stats = tiny_dataset
    rows = build.read_manifest(root / "manifests" / "train.tsv")
    st = np.cumsum([int(r["stereo"]) for r in rows])
    for n in (10, 25, 50, 100):
        assert abs(st[n - 1] / n - 0.4) <= 1.0 / n + 1e-9  # every prefix is on target
    keys = {
        s: {r["key14"] for r in build.read_manifest(root / "manifests" / f"{s}.tsv")}
        for s in build.SPLITS
    }
    assert not (keys["train"] & keys["val"]) and not (keys["train"] & keys["test"])
    assert not (keys["val"] & keys["test"])
    for r in rows:
        assert int(r["stereo"]) == has_stereo(r["smiles"])


def test_shard_dataset_and_sampler(tiny_dataset):
    from ouroboros.decode.tokenizer import SmilesTokenizer

    root, _ = tiny_dataset
    manifest = build.read_manifest(root / "manifests" / "train.tsv")
    tok = SmilesTokenizer.build(r["smiles"] for r in manifest)
    ds = ShardDataset(root / "shards", "train", tokenizer=tok)
    assert len(ds) == 100
    item = ds[3]
    assert (
        item["image"].shape == (1, 64, 64) and 0 <= item["image"].min() <= item["image"].max() <= 1
    )
    assert item["smiles"] == Chem.CanonSmiles(manifest[3]["smiles"])
    assert len(ShardDataset(root / "shards", "train", max_samples=50)) == 50
    batch = collate([ds[i] for i in range(4)], pad_id=tok.pad_id)
    assert batch["image"].shape == (4, 1, 64, 64) and batch["ids"].shape[0] == 4

    s = ResumableSampler(len(ds), seed=3)
    it = iter(s)
    full = [next(it) for _ in range(250)]  # crosses two epoch boundaries
    s2 = ResumableSampler(len(ds), seed=3)
    s2.set_position(137)
    it2 = iter(s2)
    assert [next(it2) for _ in range(113)] == full[137:]
    assert sorted(full[:100]) == list(range(100))


# ----------------------------------------------------------------------------- rotation


def test_rotate_images_conventions():
    x = torch.rand(2, 1, 16, 16)
    exact = rotate_images(x, torch.tensor([90.0, 270.0]))
    assert torch.equal(exact[0], torch.rot90(x[0], 1, dims=(-2, -1)))
    assert torch.equal(exact[1], torch.rot90(x[1], 3, dims=(-2, -1)))
    # the interpolating branch agrees with rot90 at (numerically) 90 deg
    near = rotate_images(x, torch.tensor([90.0 + 1e-7, 0.0]))
    assert torch.allclose(near[0], exact[0], atol=1e-4)
    # 45 then 45 ~ 90 away from the (zero-filled) corners
    two = rotate_images(rotate_images(x, torch.tensor([45.0, 45.0])), torch.tensor([45.0, 45.0]))
    yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    centre = ((xx + 0.5 - 8) ** 2 + (yy + 0.5 - 8) ** 2) < (4**2)
    assert (two[0, 0][centre] - exact[0, 0][centre]).abs().mean() < 0.25
    assert math.isclose(float(rotate_images(x, torch.tensor([360.0, 0.0])).sub(x).abs().max()), 0.0)
