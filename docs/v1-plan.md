# v1 plan: invariant to drawing, sensitive to chemistry

Working claim to be **tested, not assumed**: a molecular-diagram model should ignore how a molecule
is drawn (orientation, renderer, layout) while responding to edits that change its identity
(a stereocentre, a bond order, an atom type).

## Controlled arms (`scripts/hpc/v1_grid.py`)

| Arm | Trunk | Position / mixer | Rotation augmentation |
| --- | --- | --- | --- |
| Babs | plain CNN | learned absolute table | yes |
| Brel | plain CNN | distance-bias mixer (same module as C) | yes |
| C | C8 steerable CNN + group pooling | distance-bias mixer | no |

Babs→Brel isolates the position representation; Brel→C isolates equivariant convolutions against
augmentation. Residual confounds that remain (stem, pooling, parameter count) are listed in the
paper, not hidden. Optimizer parity: escnn filter coefficients are now weight-decayed like ordinary
kernels (previously ~10% of C's encoder was excluded).

## Label budget

Nested, article-sampled pools of K ∈ {0, 128, 512, 2048} independent real labels (≤10 crops per
article), mixed at one fixed ratio (0.1) with one fixed step budget. Checkpoints are selected on the
*synthetic* validation set, so a run never sees more real labels than K. Labels are engine-agreement
pseudo-labels (VERDICT), so the quantity measured is **pseudo-label efficiency** until a gold set
exists.

## Evaluation rules

- Real test = 1,170 crops (124 articles), molecule-disjoint from train/dev after InChI
  standardization; out-of-vocabulary (17) and over-length (6) labels are excluded by label property
  and counted. Report all rows **and** `n_agree ≥ 3` separately; never pool with synthetic sets.
- Uncertainty: article-level paired bootstrap on real data; seed variation reported separately;
  rotation angles are not independent samples.
- Gold standard (not yet built): 200-crop blind audit now; ~1,000–1,200 random test crops with two
  independent annotators later, plus a ~300-crop chirality stress set, reported separately.
- External systems (MolScribe, OCSRGlyph, one newer system) run on the same image list with the same
  scorer.

## Conditions that make us drop an explanation (stated in advance)

1. "Invariance at pooling time costs chirality" is dropped if a larger C (more capacity at equal
   budget) removes the gap to an orientation-preserving head, or if the gap shrinks to noise when
   parameters, optimizer and schedule are matched.
2. "Equivariance replaces augmentation" is dropped if Brel matches C across rotations *and* label
   budgets within the seed-to-seed spread.
3. If the only surviving result is B≈C on synthetic rotation, the paper is narrowed to a careful
   empirical study and the main-track claim is not made.

## Not done yet (in order)

1. Paired diagnostic sets: identity-preserving (rotation, redraw, layout) and identity-changing
   (one stereocentre / bond / atom edit, re-rendered and RDKit-verified); "both correct" and
   "responds correctly to the edit" rates on ~1,000 molecules.
2. Orientation-preserving head (D) and its depth-matched controls, and the larger-C control.
3. 200-crop blind audit sheet, then the gold test set.
4. External baselines on the same image list.
5. Power analysis for the real test size.
6. MACE stage: appendix tool only.
