"""Phase 0 smoke tests: one per subpackage, plus the heavy third-party imports."""

import torch


def test_data_canonical_smiles():
    from ouroboros.data.chem import canonical_smiles, inchikey

    assert canonical_smiles("OCC") == canonical_smiles("CCO") == "CCO"
    assert canonical_smiles("not a smiles") is None
    # stereo is kept by default and dropped on request
    assert "@" in canonical_smiles("C[C@H](N)O")
    assert "@" not in canonical_smiles("C[C@H](N)O", isomeric=False)
    assert inchikey("CCO") == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"


def test_encoder_interface():
    from ouroboros.encoder import EncoderOutput, ImageEncoder

    class Dummy(ImageEncoder):
        d_model = 8

        def forward(self, images):
            return EncoderOutput(tokens=images.flatten(2).transpose(1, 2)[..., :8])

    out = Dummy()(torch.zeros(2, 8, 4, 4))
    assert out.tokens.shape == (2, 16, 8)


def test_decode_tokenizer_roundtrip():
    from ouroboros.decode.tokenizer import SmilesTokenizer, tokenize

    smi = ["C[C@@H](Cl)/C=C/Br", "c1ccccc1O", "N#CC(=O)[O-]"]
    assert tokenize(smi[0]) == ["C", "[C@@H]", "(", "Cl", ")", "/", "C", "=", "C", "/", "Br"]
    tok = SmilesTokenizer.build(smi)
    for s in smi:
        ids = tok.encode(s)
        assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
        assert tok.decode(ids) == s


def test_train_config_overrides(tmp_path):
    from ouroboros.train.config import load_config

    (tmp_path / "base.yaml").write_text("model: {d: 4, layers: 2}\nlr: 0.1\n")
    (tmp_path / "child.yaml").write_text("base: base.yaml\nmodel: {d: 8}\n")
    cfg = load_config(tmp_path / "child.yaml", ["lr=0.5", "model.layers=6"])
    assert cfg == {"model": {"d": 8, "layers": 6}, "lr": 0.5}


def test_eval_package_imports():
    import ouroboros.eval  # noqa: F401


def test_geometry_etkdg_embed():
    from rdkit import Chem

    from ouroboros.geometry.conformers import embed_conformers

    mol = embed_conformers("C[C@H](N)C(=O)O", n_conf=3, seed=1)
    assert mol.GetNumConformers() == 3
    # stereo survives embedding
    m2 = Chem.RemoveHs(mol)
    Chem.AssignStereochemistryFrom3D(m2)
    assert Chem.MolToSmiles(m2) == Chem.CanonSmiles("C[C@H](N)C(=O)O")


def test_third_party_imports():
    import escnn.gspaces  # noqa: F401
    import mace.modules  # noqa: F401
    import webdataset  # noqa: F401
