# Related work and differences (working notes, v1)

Verification status: **V** = existence, authors and the quoted claim checked against the arXiv/venue
abstract page on 2026-10-19; **U** = taken from a reviewer's summary and not yet checked against the
full text. Scores below are from different benchmarks and must not be compared with ours.

| Work | What it does | Status | What we do differently / what is missing in it |
| --- | --- | --- | --- |
| MolScribe (Qian, Coley, Barzilay) | Predicts atoms, coordinates, bonds; rules for chirality and abbreviations | U | Symbolic structure head; we study *which* image symmetries a visual encoder may discard |
| DECIMER (Rajan, Steinbeck) | Scale (hundreds of millions of depictions), literature pipeline | U | Scale-based; no controlled prior-vs-augmentation comparison |
| MolParser (Fang et al., ICCV 2025) | E-SMILES, real data, active learning, Markush | U | Broader object coverage; not a mechanism study |
| OCSRGlyph / MarkushGlyph (arXiv 2607.28532) | Stereochemistry-aware OCSR, Markush VLM | **V** (paper exists); the "93.8% USPTO" figure is **U** | Strong external baseline to run on our test list |
| RISTER (arXiv 2608.10684) | Rotation-equivariant conv + self-attention encoder, rotation-invariant cross-attention decoder, for scene text | **V** | Architecturally close to our C. "Equivariant encoder + invariant decoder" is **not** a contribution here; only the molecular setting and the chirality-vs-invariance analysis are |
| AtomLenz (Oldenhof et al., CVPR 2024) | Atom-level OCSR with SMILES-only supervision, few-shot domain adaptation | U | "A little real data helps" is not new; we measure labels-to-accuracy with nested pools |
| MinerU.Chem (arXiv 2608.03525) | System for chemistry document parsing; 93.02% exact match on the SMILES-evaluable subset of MolRecBench-Wild (N=2,392) | **V** (93.02% on that subset); "79.66% graph accuracy" is **U** | Different denominators (subset vs all); never mix them |
| MolRecBench-Wild (OpenDataLab) | Real-document benchmark with visual/chemical difficulty labels | U | Candidate independent real test source |
| VERDICT (arXiv 2608.22183; our group) | Engine-agreement labels; chemist adjudication of 400 released PMC labels: 0.995 (3 of 4 engines), 0.958 (2 of 4) | **V** | Source of our weak-label corpus. Audit precision per tier does not cover hard, disagreed-upon structures |

## Consequences for the claims

1. Novelty is *not* "first equivariant OCSR encoder". The candidate contribution is a tested
   statement about which information a rotation-invariant representation must keep (chirality) and
   how a prior trades against augmentation and against real labels.
2. The task is bounded to single, vocabulary-covered molecule crops with full stereochemistry.
   Abbreviations, R-groups, Markush and under-specified drawings are out of scope and reported as
   such, not scored as failures.
3. Every comparison changes one thing. See `docs/v1-plan.md` for the arms and the pre-stated
   conditions under which an explanation is dropped.
