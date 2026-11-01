"""The model-free sanity set: perturbations and render labels give the documented outcomes."""

from rdkit import Chem

from ouroboros.eval.metrics import score_pair
from ouroboros.eval.sanity import GOLD, PERTURBATIONS, build_records, perturb


def _enantiomer(smi: str) -> str:
    return smi.replace("@@", "\0").replace("@", "@@").replace("\0", "@")


def _has_stereo(smi: str) -> bool:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False) != Chem.CanonSmiles(smi)


def test_gold_parses_and_is_unique():
    assert all(Chem.MolFromSmiles(s) is not None for s in GOLD)
    assert len({Chem.CanonSmiles(s) for s in GOLD}) == len(GOLD)


def test_every_perturbation_on_every_gold():
    for i, gold in enumerate(GOLD):
        for kind in PERTURBATIONS:
            s = score_pair(perturb(gold, kind, seed=i), gold)
            if kind in ("random_smiles", "salt"):
                assert s.exact, (kind, gold)
            elif kind == "no_stereo":
                assert s.exact_flat and s.exact == (not _has_stereo(gold)), gold
            elif kind == "add_methyl":
                assert s.valid and not s.exact_flat, gold
            else:  # syntax_error, blank
                assert not s.valid, (kind, gold)


def test_render_and_mirror_labels():
    gold = ["C[C@H](N)C(=O)O", "C/C=C/C(=O)O", "O[C@H]1CC[C@@H](O)CC1"]
    rows = build_records(gold=gold, size=128)
    assert len(rows) == 3 * len(gold)
    by = {(r["subset"], r["gold"]): r["pred"] for r in rows}
    for g in gold:
        assert score_pair(by[("render", g)], g).exact, g
        m = score_pair(by[("mirror", g)], g)
        achiral = Chem.CanonSmiles(_enantiomer(g)) == Chem.CanonSmiles(g)
        assert m.exact_flat and m.exact == achiral, g
    perturbed = [r for r in rows if r["subset"] == "perturb"]
    assert [r["perturbation"] for r in perturbed] == list(PERTURBATIONS[: len(gold)])
