# Exhaustive Verified Literature Review for Application 2 (`002-mlip-blindspot-distill`)
## Literature-Driven Modern Medicinal-Chemistry Bioisostere Blind-Spot Discovery & Active Distillation for Universal MLIPs

**Total Verified Papers:** 42 peer-reviewed / arXiv publications across 5 thematic clusters  
**Verification Protocol:** All authors, publication years, venues, DOIs/arXiv IDs, quantum-chemical levels of theory, element/charge constraints, and dataset compositions verified via web search.

---

## Executive Alignment Summary: Exact Constraints of MACE-OFF23 & Organic MLIP Corpora

Before cataloging the 42 papers, any active distillation pipeline targeting **MACE-OFF23** (and cross-auditing against **AIMNet2** and **ANI-2x**) must strictly enforce the following physical and quantum-chemical alignment invariants:

| Model / Dataset | Supported Elements | Charge & Spin Constraints | Reference QM Level of Theory (Code) | Key Training Composition & Omissions |
| :--- | :--- | :--- | :--- | :--- |
| **MACE-OFF23** (Kovács et al., *JACS* 2025) | **10 elements:** `H, C, N, O, F, P, S, Cl, Br, I` *(omits B, Si, Se, metals)* | **Strictly neutral (`Q = 0`), closed-shell singlet (`2S+1 = 1`) only.** Short-range cutoff; no charge/spin inputs or long-range Coulomb term. | **$\omega$B97M-D3(BJ) / def2-TZVPPD** (gas phase, **Psi4**) | ~85% neutral subset of SPICE v1 ($\le 50$ atoms) + **1,597** recomputed QMugs conformers (51–90 atoms) + **2,748** water clusters (**0** tripeptides in train; 144 in test). Strips all ions, salts, and charged groups. |
| **AIMNet2** (Anstine et al., *Chem. Sci.* 2025) | **14 elements:** `H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I` | **Neutral, cationic, anionic, & zwitterionic** (`Q` passed explicitly via Neural Spin-Charge Equilibration + damped Coulomb + D3BJ). | **$\omega$B97M-D3(BJ) / def2-TZVPP** *(lacks diffuse `D`!)* (**ORCA 5**) | ~20M hybrid DFT configurations across neutral & charged druglike/element-organic space. **Cannot** mix raw energies directly with MACE-OFF23 due to `def2-TZVPP` vs `def2-TZVPPD` basis offset! |
| **ANI-2x** (Devereux et al., *JCTC* 2020) | **7 elements:** `H, C, N, O, F, S, Cl` *(omits P, Br, I, B, Si)* | **Strictly neutral (`Q = 0`), closed-shell singlet.** Local Behler–Parrinello AEVs. | **$\omega$B97X / 6-31G(d)** (**Gaussian**) | ~8.9M conformations + torsional refinement scans; pre-2020 GDB/ChEMBL small-molecule chemical space. |
| **SPICE v1 / v2** (Eastman et al., 2023 / 2024) | v1: 15 elements; v2 adds **`B, Si`** (17 total) | Neutral & charged small molecules, dipeptides, amino-acid dimers, solvated shells. | **$\omega$B97M-D3(BJ) / def2-TZVPPD** (**Psi4**) | Drawn from legacy PubChem & biomolecules; severely under-samples post-2020 strained 3D medchem cages (BCHs, 2-oxa-BCHs, BCHeps, cubanes, $\text{SF}_5$-cages). |
| **QMugs** (Isert et al., *Sci. Data* 2022) | `H, C, N, O, F, P, S, Cl, Br, I` | 665k ChEMBL27 molecules (~2M conformers), including charged species. | Geometries: **GFN2-xTB**; single-point: **$\omega$B97X-D / def2-SVP** (**Psi4**) | Pre-2021 ChEMBL27 only; lower basis/functional (`def2-SVP`), requiring fullPsi4 recomputation before MACE-OFF23 ingestion. |
| **OMol25** (Levine et al., Meta FAIR 2025) | **83 elements** | Variable charges $Q$ and spin multiplicities $2S+1$; up to 350 atoms. | **$\omega$B97M-V / def2-TZVPD** (**ORCA 6.0.1**) | >100M DFT calculations, but uses VV10 nonlocal dispersion ($\omega$B97M-V) and single-diffuse `def2-TZVPD`—incompatible for naive single-head mixing with MACE-OFF23 without multi-head replay or $\Delta$-learning. |

---

## Cluster 2A — Universal / Foundation Organic MLIPs (10 Papers)

### [1] MACE Architecture
* **Authors & Year:** Ilyes Batatia, Dávid Péter Kovács, Gregor N. C. Simm, Christoph Ortner, and Gábor Csányi (2022)
* **Title:** *MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields*
* **Venue / Identifier:** *Advances in Neural Information Processing Systems (NeurIPS)*, 35, 11423–11436 (2022); `arXiv:2206.07697`
* **Core Contribution:** Combines $E(3)$-equivariant message passing with body-ordered Atomic Cluster Expansion (ACE) tensor products, achieving arbitrary body-order four-body+ local representations in just 2 message-passing layers (`L=2`). Delivers steep learning curves and fast GPU parallelization on rMD17, 3BPA, and AcAc.
* **Key Limitation Relative to Our Work:** Purely architectural; relies entirely on the coverage of training data for strained four-body angular/transannular correlations and includes no intrinsic single-model epistemic uncertainty head or out-of-distribution (OOD) chemical-space discovery mechanism.

### [2] MACE-OFF23 Transferable Organic Force Field
* **Authors & Year:** Dávid Péter Kovács, J. Harry Moore, Nicholas J. Browning, Ilyes Batatia, Joshua T. Horton, Yixuan Pu, Venkat Kapil, William C. Witt, Ioan-Bogdan Magdău, Daniel J. Cole, and Gábor Csányi (2025; preprint Dec 2023)
* **Title:** *MACE-OFF: Short-Range Transferable Machine Learning Force Fields for Organic Molecules*
* **Venue / Identifier:** *Journal of the American Chemical Society (JACS)*, 147(21), 17598–17611 (2025); DOI: `10.1021/jacs.4c07099`; `arXiv:2312.15211`
* **Core Contribution:** Introduces MACE-OFF23 (small, medium, large), a purely short-range transferable organic MLIP covering **10 elements (`H, C, N, O, F, P, S, Cl, Br, I`)** trained exclusively on **charge-neutral, closed-shell** configurations computed at **$\omega$B97M-D3(BJ)/def2-TZVPPD** in Psi4 (~85% neutral SPICE v1 + 1,597 recomputed 51–90 atom QMugs conformers + 2,748 water clusters; tested zero-shot on 144 COMP6 tripeptides, TorsionNet-500, liquid densities, and crystal lattice energies).
* **Key Limitation Relative to Our Work:** (1) Strictly restricted to neutral closed-shell 10-element molecules (cannot process salts, ions, B, or Si); (2) contains only 1,597 structures >50 atoms from pre-2021 ChEMBL/QMugs and legacy PubChem/SPICE, leaving modern post-2020 3D strained bioisosteres (BCHs, 2-oxa-BCHs, BCHeps, cubanes, $\text{SF}_5$-cages) systematically under-represented; (3) large/medium models remain costly for million-conformer MD/screening without student distillation.

### [3] ANI-1 Potential
* **Authors & Year:** Justin S. Smith, Olexandr Isayev, and Adrian E. Roitberg (2017)
* **Title:** *ANI-1: an extensible neural network potential with DFT accuracy at force field computational cost*
* **Venue / Identifier:** *Chemical Science*, 8(4), 3192–3203 (2017); DOI: `10.1039/C6SC05720A`
* **Core Contribution:** Pioneered transferable deep neural network potentials for organic molecules using modified Behler–Parrinello Atomic Environment Vectors (AEVs) trained on ~20M off-equilibrium normal-mode conformations of 57,462 small GDB-11 molecules (`H, C, N, O`) at $\omega$B97X/6-31G(d).
* **Key Limitation Relative to Our Work:** Restricted to 4 elements (`H, C, N, O`), lacks higher-order equivariant angular resolution for acute cage bond angles ($60^\circ$–$90^\circ$), and relies on normal-mode sampling of tiny $\le 8$-heavy-atom GDB molecules rather than real medicinal-chemistry scaffolds.

### [4] ANI-2x Potential
* **Authors & Year:** Christian Devereux, Justin S. Smith, Kate K. Huddleston, Kipton Barros, Roman Zubatyuk, Olexandr Isayev, and Adrian E. Roitberg (2020)
* **Title:** *Extending the Applicability of the ANI Deep Learning Molecular Potential to Sulfur and Halogens*
* **Venue / Identifier:** *Journal of Chemical Theory and Computation (JCTC)*, 16(7), 4192–4202 (2020); DOI: `10.1021/acs.jctc.0c00121`
* **Core Contribution:** Extends ANI via active learning and torsional refinement to **7 elements (`H, C, N, O, F, S, Cl`)** across ~8.9M $\omega$B97X/6-31G(d) structures, providing a widely deployed 8-model ensemble for neutral drug-like molecules.
* **Key Limitation Relative to Our Work:** Omits `P, Br, I` (excluding brominated/iodinated intermediates and phosphonate/phosphine-oxide bioisosteres), uses double-zeta `6-31G(d)` DFT (which poorly describes hypervalent sulfur $d$-polarization in $\text{SF}_5$ and sulfoximines!), and predates modern 2020–2025 bicyclic bioisostere chemistry.

### [5] AIMNet2 Potential
* **Authors & Year:** Dylan M. Anstine, Roman Zubatyuk, and Olexandr Isayev (2025)
* **Title:** *AIMNet2: a neural network potential to meet your neutral, charged, organic, and elemental-organic needs*
* **Venue / Identifier:** *Chemical Science*, 16(23), 10228–10244 (2025); DOI: `10.1039/D4SC08572H`
* **Core Contribution:** Covers **14 elements (`H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I`)** across **neutral, cationic, anionic, and zwitterionic** states by combining message-passing short-range interactions with Neural Spin-Charge Equilibration (NSC), explicit Coulomb electrostatics, and D3(BJ) dispersion, trained on ~20M **$\omega$B97M-D3(BJ)/def2-TZVPP** (ORCA) calculations.
* **Key Limitation Relative to Our Work:** Uses `def2-TZVPP` (no diffuse `D` functions) rather than MACE-OFF23's `def2-TZVPPD`, creating a systematic basis-set energy offset; uses lower-body-order message passing than MACE; and remains trained on static database chemical space rather than continuously mined patent/literature bioisostere motifs.

### [6] NequIP Architecture
* **Authors & Year:** Simon Batzner, Albert Musaelian, Lixin Sun, Mario Geiger, Jonathan P. Mailoa, Mordechai Kornbluth, Nicola Molinari, Tess E. Smidt, and Boris Kozinsky (2022)
* **Title:** *E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials*
* **Venue / Identifier:** *Nature Communications*, 13, 2453 (2022); DOI: `10.1038/s41467-022-29939-5`
* **Core Contribution:** Demonstrated that $E(3)$-equivariant Clebsch–Gordan tensor products over spherical harmonics ($l \le 3$) achieve dramatic sample-efficiency gains and lower force MAEs than invariant MPNNs/kernels on MD17 and liquid water.
* **Key Limitation Relative to Our Work:** Deep 2-body message passing (4–6 layers) expands the effective receptive field and slows MD inference; originally trained as single-system potentials without cross-chemical organic transferability or OOD bioisostere discovery.

### [7] Allegro Strictly Local Equivariant Potential
* **Authors & Year:** Albert Musaelian, Simon Batzner, Anders Johansson, Lixin Sun, Cameron J. Owen, Mordechai Kornbluth, and Boris Kozinsky (2023)
* **Title:** *Learning local equivariant representations for large-scale atomistic dynamics*
* **Venue / Identifier:** *Nature Communications*, 14, 579 (2023); DOI: `10.1038/s41467-023-36329-y`
* **Core Contribution:** Eliminates graph message passing entirely in favor of strictly local, ordered equivariant tensor-product layers on central-atom neighbor pairs, achieving $O(N)$ spatial decomposition and 100M-atom GPU scaling while retaining NequIP-class accuracy.
* **Key Limitation Relative to Our Work:** Strict locality without message passing makes it an ideal fast **student** architecture candidate for distillation, yet out-of-the-box it lacks pre-trained foundation weights for 10-element medicinal chemistry or active blind-spot curation.

### [8] EquiformerV2 Equivariant Transformer
* **Authors & Year:** Yi-Lun Liao, Brandon M. Wood, Abhishek Das, and Tess Smidt (2024)
* **Title:** *EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations*
* **Venue / Identifier:** *International Conference on Learning Representations (ICLR)* (2024); `arXiv:2306.12059`
* **Core Contribution:** Replaces $SO(3)$ Clebsch–Gordan bottlenecks with $e$-SCN convolutions, attention re-normalization, separable $S^2$ activations, and separable layer normalization, scaling equivariant Transformers up to $L_{\max}=6$ on OC20/OC22 and OMol25.
* **Key Limitation Relative to Our Work:** High parameter count (31M–153M) and computational latency make direct deployment in high-throughput conformational/torsional screening of patent libraries expensive without targeted knowledge distillation into compact force fields.

### [9] Open Molecules 2025 (OMol25)
* **Authors & Year:** Daniel S. Levine, Muhammed Shuaibi, Evan Walter Clark Spotte-Smith, Michael G. Taylor, Muhammad R. Hasyim, Kyle Michel, Ilyes Batatia, Gábor Csányi, Misko Dzamba, Peter Eastman, ..., Samuel M. Blau, and Brandon M. Wood (2025)
* **Title:** *The Open Molecules 2025 (OMol25) Dataset, Evaluations, and Models*
* **Venue / Identifier:** Meta FAIR Preprint, `arXiv:2505.08762` (2025)
* **Core Contribution:** Largest open molecular DFT corpus to date (>100M calculations across ~83M systems, **83 elements**, up to 350 atoms, variable charge/spin) computed at **$\omega$B97M-V/def2-TZVPD** in ORCA 6.0.1, paired with massive eSEN/EquiformerV2/MACE baseline models.
* **Key Limitation Relative to Our Work:** Computed at $\omega$B97M-V/def2-TZVPD (VV10 nonlocal dispersion, single-diffuse basis) rather than SPICE/MACE-OFF23's $\omega$B97M-D3(BJ)/def2-TZVPPD; massive compute footprint motivates lightweight domain-distilled organic MLIPs; still relies on archived public databases rather than visual OCSR extraction from emerging medicinal chemistry patents/papers.

### [10] MACE-MP-0 & Universal Materials Foundation Models
* **Authors & Year:** Ilyes Batatia, Philipp Benner, Yuan Chiang, Alin M. Elena, Dávid P. Kovács, Janosh Riebesell, ..., Kristin A. Persson, and Gábor Csányi (2024)
* **Title:** *A foundation model for atomistic materials chemistry*
* **Venue / Identifier:** `arXiv:2401.00096` (2024) *(complemented in speed-focused foundation design by Mark Neumann et al., "Orb: A Fast, Scalable Neural Network Potential", `arXiv:2410.22570`, 2024)*
* **Core Contribution:** Demonstrates that a single higher-order equivariant MACE potential trained on MPtrj (PBE+U inorganic crystals across 89 elements) stabilizes out-of-the-box MD across solids, liquids, surfaces, and gas-phase molecules, and serves as a universal pre-trained backbone for few-shot transfer learning.
* **Key Limitation Relative to Our Work:** PBE-level inorganic DFT severely underpredicts organic dispersion, conformational isomerization, and strained-ring torsional barriers (~5–15 kcal/mol errors on drug-like torsions), proving why hybrid meta-GGA organic foundation models (MACE-OFF23) and targeted organic fine-tuning are indispensable.

---

## Cluster 2B — Organic Quantum Chemistry Datasets & Force-Field Torsion Benchmarks (8 Papers)

### [11] SPICE Dataset (v1)
* **Authors & Year:** Peter Eastman, Pavan Kumar Behara, David L. Dotson, Raimondas Galvelis, John E. Herr, Josh T. Horton, Yuezhi Mao, John D. Chodera, Benjamin P. Pritchard, Yuanqing Wang, Gianni De Fabritiis, and Thomas E. Markland (2023)
* **Title:** *SPICE, A Dataset of Drug-like Molecules and Peptides for Training Machine Learning Potentials*
* **Venue / Identifier:** *Scientific Data*, 10, 11 (2023); DOI: `10.1038/s41597-022-01882-6`
* **Core Contribution:** Establishes the community gold-standard organic/biomolecular MLIP training corpus: 1.1M high- and low-energy MD/conformer configurations across dipeptides, solvated amino acids, PubChem small molecules ($\le 50$ atoms), and dimers at **$\omega$B97M-D3(BJ)/def2-TZVPPD** (Psi4).
* **Key Limitation Relative to Our Work:** Small-molecule subset was drawn randomly from legacy PubChem ($\le 50$ atoms) and sampled via classical/high-$T$ MD rather than systematic 1D/2D dihedral barrier scans of modern strained 3D bioisosteres (BCHs, BCHeps, 2-oxa-BCHs, cubanes, $\text{SF}_5$-aryls).

### [12] SPICE 2 & Nutmeg Potentials
* **Authors & Year:** Peter Eastman, Benjamin P. Pritchard, John D. Chodera, and Thomas E. Markland (2024)
* **Title:** *Nutmeg and SPICE: Models and Data for Biomolecular Machine Learning*
* **Venue / Identifier:** *Journal of Chemical Theory and Computation (JCTC)*, 20(19), 8583–8593 (2024); DOI: `10.1021/acs.jctc.4c00794`
* **Core Contribution:** Doubles SPICE to **SPICE v2** (~2M conformations) by adding >13,000 new PubChem molecules, 194k ligand–amino-acid dimers, 1,000 water clusters, 1,397 water-solvated PubChem shells, and **2 new elements (`B` and `Si`)**; introduces **Nutmeg** (TensorNet injected with RDKit Gasteiger charges for polar/charged molecules).
* **Key Limitation Relative to Our Work:** MACE-OFF23 was frozen prior to SPICE v2's `B`/`Si` and expanded PubChem additions; moreover, SPICE v2 still relies on random PubChem sampling rather than uncertainty-driven active mining of newly disclosed medicinal-chemistry patent/literature bioisosteres.

### [13] QMugs Dataset
* **Authors & Year:** Clemens Isert, Kenneth Atz, José Jiménez-Luna, and Gisbert Schneider (2022)
* **Title:** *QMugs, quantum mechanical properties of drug-like molecules*
* **Venue / Identifier:** *Scientific Data*, 9, 273 (2022); DOI: `10.1038/s41597-022-01390-7`
* **Core Contribution:** Provides ~2M conformers across >665,000 bioactive ChEMBL27 drug-like molecules (up to 100 heavy atoms; `H, C, N, O, F, P, S, Cl, Br, I`) with GFN2-xTB geometries and $\omega$B97X-D/def2-SVP wavefunctions/energetics.
* **Key Limitation Relative to Our Work:** (1) Only 3 equilibrium conformers per molecule (no high-energy torsional barriers or off-equilibrium force trajectories!); (2) `def2-SVP` basis is incompatible with MACE-OFF23 without full $\omega$B97M-D3(BJ)/def2-TZVPPD recomputation; (3) frozen at pre-2021 ChEMBL27, missing 2021–2026 caged bioisostere scaffolds.

### [14] GEOM Dataset
* **Authors & Year:** Simon Axelrod and Rafael Gómez-Bombarelli (2022)
* **Title:** *GEOM, energy-annotated molecular conformations for property prediction and molecular generation*
* **Venue / Identifier:** *Scientific Data*, 9, 185 (2022); DOI: `10.1038/s41597-022-01288-4`
* **Core Contribution:** Supplies >37M CREST/GFN2-xTB conformer ensembles across 450,000 molecules (QM9 + ~304k AICures/ChEMBL drug-like compounds in GEOM-Drugs), powering 3D generative conformer models.
* **Key Limitation Relative to Our Work:** Almost entirely semi-empirical GFN2-xTB labels (known to distort strained small-ring puckering and hypervalent $\text{S(VI)}$ / $\text{SF}_5$ rotational barriers relative to hybrid triple-zeta DFT) and restricted to local minima rather than full torsional potential energy curves.

### [15] ANI-1ccx & COMP6 Transferability Benchmark
* **Authors & Year:** Justin S. Smith, Benjamin T. Nebgen, Roman Zubatyuk, Nicholas Lubbers, Christian Devereux, Kipton Barros, Sergei Tretiak, Olexandr Isayev, and Adrian E. Roitberg (2019)
* **Title:** *Approaching coupled cluster accuracy with a general-purpose neural network potential through transfer learning*
* **Venue / Identifier:** *Nature Communications*, 10, 2903 (2019); DOI: `10.1038/s41467-019-10827-4`
* **Core Contribution:** Demonstrates transfer learning of ANI from 5M DFT structures to ~500k $\text{CCSD(T)*/CBS}$ data points (`H, C, N, O`), validated on the **COMP6** out-of-sample benchmark (GDB7to9, GDB10to13, DrugBank, Tripeptides, S66x8, ANI-MD).
* **Key Limitation Relative to Our Work:** Restricted to 4 elements (`H, C, N, O`); COMP6's DrugBank subset reflects legacy flat/aromatic drugs rather than modern fluorinated/heteroatom-rich $sp^3$-caged bioisosteres, motivating a dedicated **Bioisostere-Torsion & Strain Blind-Spot Benchmark** alongside COMP6 retention checks.

### [16] nablaDFT & $\nabla^2$DFT Benchmarks
* **Authors & Year:** Kuzma Khrabrov, Anton Ber, Artem Tsypin, Konstantin Ushenin, Egor Rumiantsev, Alexander Telepov, Dmitry Protasov, Ilya Shenbin, Anton Alekseev, Evgeny Shirokikh, Sergey Nikolenko, Elena Tutubalina, and Artur Kadurin (2024; extending Khrabrov et al., *PCCP* 2022)
* **Title:** *$\nabla^2$DFT: A Universal Quantum Chemistry Dataset of Drug-Like Molecules and a Benchmark for Neural Network Potentials*
* **Venue / Identifier:** *NeurIPS Datasets and Benchmarks* (2024), `arXiv:2406.14347` *(extending `nablaDFT`, Phys. Chem. Chem. Phys., 24, 25853–25863, 2022)*
* **Core Contribution:** Expands MOSES drug-like molecular space to ~1.93M molecules and >16M conformations + relaxation trajectories + Kohn–Sham Hamiltonians at $\omega$B97X-D/def2-SVP, systematically benchmarking equivariant NNPs across unseen scaffolds.
* **Key Limitation Relative to Our Work:** Sourced from MOSES (filtered ZINC clean leads biased toward classical 2D medicinal chemistry rules, explicitly excluding many strained polycyclic cages, hypervalent $\text{SF}_5$, and heavy halogens) and computed at small-basis `def2-SVP`.

### [17] TorsionNet & TorsionNet-500 Benchmark
* **Authors & Year:** Brajesh K. Rai, Vishnu Sresht, Qingyi Yang, Ray Unwalla, Meihua Tu, Alan M. Mathiowetz, and Gregory A. Bakken (2022)
* **Title:** *TorsionNet: A Deep Neural Network to Rapidly Predict Small-Molecule Torsional Energy Profiles with the Accuracy of Quantum Mechanics*
* **Venue / Identifier:** *Journal of Chemical Information and Modeling (JCIM)*, 62(4), 785–800 (2022); DOI: `10.1021/acs.jcim.1c01346`
* **Core Contribution:** Built at Pfizer from >1M B3LYP/6-31G\*\* torsion points; releases the open **TorsionNet-500** benchmark of 500 diverse pharmaceutically relevant 1D torsion profiles (24 grid points $\times 15^\circ$ each), now a standard benchmark in MACE-OFF23 and AIMNet2 papers.
* **Key Limitation Relative to Our Work:** TorsionNet-500 centers on classical rotatable biaryl/amide/heteroaryl linkages (`H, C, N, O, F, S, Cl`) and lacks bridgehead exit-vector torsions attached to bicyclo[1.1.1]pentanes, bicyclo[2.1.1]hexanes, 2-oxabicyclo[2.1.1]hexanes, cubanes, or $\text{SF}_5$ rotors.

### [18] OpenFF 2.0.0 ("Sage") & Drug-Fragment Torsion Strain Benchmarks
* **Authors & Year:** Simon Boothroyd, Pavan Kumar Behara, Owen C. Madin, David F. Hahn, Hyesu Jang, Vytautas Gapsys, ..., John D. Chodera, Michael R. Shirts, and David L. Mobley (2023)
* **Title:** *Development and Benchmarking of Open Force Field 2.0.0: The Sage Small Molecule Force Field*
* **Venue / Identifier:** *Journal of Chemical Theory and Computation (JCTC)*, 19(11), 3251–3275 (2023); DOI: `10.1021/acs.jctc.3c00039` *(paired with Sellers, James & Gobbi, JCIM 2017, 57, 1265–1275 on 62 CCSD(T)/CBS fragment torsion scans, and Lahey, Phuc & Rowley, JCIM 2020, 60, 6258–6268 benchmarking ANI-2x vs MM on biaryl drug torsions)*
* **Core Contribution:** Fits SMIRNOFF direct-chemical-perception valence/torsion and condensed-phase LJ parameters against QCArchive $\omega$B97M-D3(BJ)/def2-TZVP torsion drives and liquid properties, establishing standard **RMSE, barrier height error ($\Delta\text{BH}$), and Torsion Fingerprint Deviation (TFD)** evaluation protocols.
* **Key Limitation Relative to Our Work:** Fixed cosine-series MM functional forms fail on coupled pyramidalization/conjugation/transannular steric clashes in strained cages, while MLIP evaluations on Sellers/Rowley/OpenFF torsion sets have never targeted literature-mined post-2020 3D bioisosteres.

---

## Cluster 2C — Active Learning & Uncertainty Quantification in MLIPs (10 Papers)

### [19] Query-by-Committee Active Learning for Chemical Space (ANI-1x)
* **Authors & Year:** Justin S. Smith, Ben Nebgen, Nicholas Lubbers, Olexandr Isayev, and Adrian E. Roitberg (2018)
* **Title:** *Less is more: Sampling chemical space with active learning*
* **Venue / Identifier:** *The Journal of Chemical Physics*, 148(24), 241733 (2018); DOI: `10.1063/1.5023802`; `arXiv:1801.09319`
* **Core Contribution:** Introduces automated Query-by-Committee (QBC) active learning across chemical space using normalized ensemble standard deviation $\rho = \sigma_E / \sqrt{N_{\text{atoms}}}$, building the 5M-structure ANI-1x dataset that outperforms 20M random ANI-1 points with 75% fewer QM calculations and introducing COMP6.
* **Key Limitation Relative to Our Work:** Requires training and running $M=8$ independent models per active-learning cycle ($8\times$ cost for large equivariant foundation models like MACE-OFF23) and selects candidates only from static pre-enumerated databases rather than visual literature/patent mining.

### [20] DP-GEN Concurrent Learning Platform
* **Authors & Year:** Yuzhi Zhang, Haidi Wang, Weijie Chen, Jinzhe Zeng, Linfeng Zhang, Han Wang, and Weinan E (2020)
* **Title:** *DP-GEN: A concurrent learning platform for the generation of reliable deep learning based potential energy models*
* **Venue / Identifier:** *Computer Physics Communications*, 253, 107206 (2020); DOI: `10.1016/j.cpc.2020.107206`
* **Core Contribution:** Formalizes the closed-loop **Explore $\to$ Label $\to$ Train** paradigm using maximum atomic force deviation $\epsilon_t = \max_i \sqrt{\langle \|\mathbf{F}_{i,m} - \bar{\mathbf{F}}_i\|^2 \rangle}$ inside an acceptance window $[\sigma_{\text{lo}}, \sigma_{\text{hi}}]$ to filter out both redundant low-error frames and unphysical atomic-overlap crashes.
* **Key Limitation Relative to Our Work:** Designed for single-material/alloy MD exploration from scratch rather than chemical-space scaffold discovery or catastrophic-forgetting-safe distillation of pre-trained organic foundation models.

### [21] Committee Neural Network Potentials
* **Authors & Year:** Christoph Schran, Krystof Brezina, and Ondrej Marsalek (2020)
* **Title:** *Committee neural network potentials control generalization errors and enable active learning*
* **Venue / Identifier:** *The Journal of Chemical Physics*, 153(10), 104105 (2020); DOI: `10.1063/5.0016004`
* **Core Contribution:** Shares atomic descriptor computation across committee members to cut ensemble overhead and introduces committee-disagreement restraining bias potentials to prevent MD trajectories from entering catastrophic extrapolation Holes.
* **Key Limitation Relative to Our Work:** Shared fixed Behler–Parrinello descriptors do not directly carry over to learned multi-layer equivariant MACE representations unless implemented via last-layer/multi-head shallow ensembles or gradient kernels.

### [22] FLARE On-the-Fly Bayesian Force Fields
* **Authors & Year:** Jonathan Vandermause, Steven B. Torrisi, Simon Batzner, Yu Xie, Lixin Sun, Alexie M. Kolpak, and Boris Kozinsky (2020)
* **Title:** *On-the-fly active learning of interpretable Bayesian force fields for atomistic rare events*
* **Venue / Identifier:** *npj Computational Materials*, 6, 20 (2020); DOI: `10.1038/s41524-020-0283-z`
* **Core Contribution:** Uses exact Gaussian Process (GP) predictive variance over 2+3-body local environments to trigger DFT calls on-the-fly during rare-event MD (diffusion, reactions) and maps converged GPs onto fast spline tables.
* **Key Limitation Relative to Our Work:** Exact GPs scale cubically $O(N_{\text{sparse}}^3)$ and struggle across combinatorial 10-element medicinal-chemistry space compared to deep higher-order equivariant foundation networks.

### [23] D-Optimality Active Learning for Moment Tensor Potentials
* **Authors & Year:** Evgeny V. Podryabinkin and Alexander V. Shapeev (2017)
* **Title:** *Active learning of linearly parametrized interatomic potentials*
* **Venue / Identifier:** *Computational Materials Science*, 140, 171–180 (2017); DOI: `10.1016/j.commatsci.2017.08.031`
* **Core Contribution:** Introduces the **D-optimality extrapolation grade $\gamma(\mathbf{x})$** via the maximum-volume (`MaxVol`) submatrix of the Jacobian $\partial E / \partial \theta$, guaranteeing mathematically that $\gamma > 1$ flags geometric extrapolation outside the convex/linear span of training environments using **only a single model**.
* **Key Limitation Relative to Our Work:** Formulated for linearly parameterized MTPs (~hundreds of parameters); applying Jacobian D-optimality to million-parameter equivariant GNNs requires sketched/last-layer neural tangent features (see [24]).

### [24] Single-Model Neural Gradient Features & LCMD Batch Active Learning
* **Authors & Year:** Viktor Zaverkin, David Holzmüller, Ingo Steinwart, and Johannes Kästner (2022)
* **Title:** *Exploring chemical and conformational spaces by batch mode deep active learning*
* **Venue / Identifier:** *Digital Discovery*, 1(5), 605–620 (2022); DOI: `10.1039/D2DD00034B` *(extended to Uncertainty-Biased MD in Zaverkin et al., npj Comput. Mater., 10, 83, 2024)*
* **Core Contribution:** Eliminates ensembles entirely by combining **sketched/last-layer neural network gradient features** (Neural Tangent Kernel posterior covariance) with **Largest Cluster Maximum Distance (LCMD)** batch selection—simultaneously maximizing epistemic uncertainty AND batch structural diversity across both chemical composition and 3D conformers.
* **Key Limitation Relative to Our Work:** Evaluated on small-molecule benchmarks trained from scratch rather than auditing pre-trained universal organic foundation models (MACE-OFF23) on newly mined medicinal-chemistry scaffolds. **Directly inspires our ensemble-free LCMD selection engine!**

### [25] Uncertainty-Driven Dynamics for Active Learning (UDD-AL)
* **Authors & Year:** Maksim Kulichenko, Kipton Barros, Nicholas Lubbers, Ying Wai Li, Richard Messerly, Sergei Tretiak, Justin S. Smith, and Benjamin Nebgen (2023)
* **Title:** *Uncertainty-driven dynamics for active learning of interatomic potentials*
* **Venue / Identifier:** *Nature Computational Science*, 3(3), 230–239 (2023); DOI: `10.1038/s43588-023-00406-5`
* **Core Contribution:** Adds a differentiable Gaussian bias potential favoring high ensemble uncertainty $E_{\text{bias}}(\sigma^2)$ directly into MD forces, driving rapid barrier crossing into informative high-uncertainty transition/conformational regions without unphysical high temperatures.
* **Key Limitation Relative to Our Work:** Relies on multi-model committee variance and explores conformational barriers of *already chosen* molecules rather than discovering *which* novel medicinal-chemistry scaffolds harbor blind spots in the first place.

### [26] Fast Single-Model Latent GMM Uncertainty for Equivariant MLIPs
* **Authors & Year:** Albert Zhu, Simon Batzner, Albert Musaelian, and Boris Kozinsky (2023)
* **Title:** *Fast uncertainty estimates in deep learning interatomic potentials*
* **Venue / Identifier:** *The Journal of Chemical Physics*, 158(16), 164111 (2023); DOI: `10.1063/5.0136574`
* **Core Contribution:** Fits a lightweight **Gaussian Mixture Model (GMM) negative log-likelihood** directly on the per-atom equivariant scalar latent embeddings of a **single** frozen NequIP/Allegro network, matching 10-model deep ensemble force-error Spearman correlation at ~1$\times$ inference cost.
* **Key Limitation Relative to Our Work:** Pure density estimation in frozen pre-trained latent space can suffer from "feature collapse" if two distinct strained cages map nearby; combining **latent Mahalanobis/GMM density [26]** with **cross-family disagreement (MACE-OFF23 vs AIMNet2) + last-layer gradient LCMD [24]** cures single-signal blind spots!

### [27] Direct Propagation of Shallow Ensembles
* **Authors & Year:** Matthias Kellner and Michele Ceriotti (2024)
* **Title:** *Uncertainty quantification by direct propagation of shallow ensembles*
* **Venue / Identifier:** *Machine Learning: Science and Technology*, 5(3), 035006 (2024); DOI: `10.1088/2632-2153/ad594a`
* **Core Contribution:** Attaches $K$ lightweight linear/MLP readout heads ("shallow ensemble") to a **single shared equivariant backbone** trained with calibrated negative log-likelihood (NLL) loss, delivering calibrated per-atom force/energy uncertainties and analytical error propagation into thermodynamic observables at <5% overhead.
* **Key Limitation Relative to Our Work:** Requires modifying/fine-tuning the readout head first (cannot score a strictly frozen checkpoint prior to readout fitting unless paired with latent distance [26] or cross-model committees). **Ideal architecture for our distilled MACE student model so the released potential ships with built-in calibrated UQ!**

### [28] Quantitative Latent-Space Distance Uncertainty Metric
* **Authors & Year:** Jon Paul Janet, Chenru Duan, Tzuhsiung Yang, Aditya Nandy, and Heather J. Kulik (2019)
* **Title:** *A quantitative uncertainty metric controls error in neural network-driven chemical discovery*
* **Venue / Identifier:** *Chemical Science*, 10(34), 7913–7922 (2019); DOI: `10.1039/C9SC02298H`
* **Core Contribution:** Establishes calibrated latent-space nearest-neighbor distance (calibrated to predictive error residuals via heteroscedastic variance modeling) as an inexpensive, architecture-agnostic OOD gate that outperforms raw ensemble std for chemical-space extrapolation.
* **Key Limitation Relative to Our Work:** Applied to static molecular/complex scalar property prediction rather than per-atom equivariant force fields and continuous dihedral potential energy surfaces.

---

## Cluster 2D — Modern 3D Medicinal-Chemistry Bioisosteres & Caged Scaffolds (8 Papers)

### [29] "Escape from Flatland" ($F_{\text{sp}^3}$)
* **Authors & Year:** Frank Lovering, Jack Bikker, and Christine Humblet (2009)
* **Title:** *Escape from Flatland: Increasing Saturation as an Approach to Improving Clinical Success*
* **Venue / Identifier:** *Journal of Medicinal Chemistry*, 52(21), 6752–6756 (2009); DOI: `10.1021/jm901241e`
* **Core Contribution:** Demonstrates across clinical progression cohorts (Discovery $\to$ Phase I–III $\to$ Drugs) that higher fraction of $sp^3$ carbons ($F_{\text{sp}^3}$) and stereochemical complexity lower crystal packing/melting points, boost aqueous solubility, and reduce CYP450/clinical attrition—triggering the modern 3D bioisostere revolution.
* **Key Limitation Relative to Our Work:** Motivates *why* medicinal chemistry shifted rapidly into strained $sp^3$ polycycles after 2015, creating a growing distribution shift between contemporary pharma patents and legacy flat-aromatic QM datasets (GDB/MOSES/early ChEMBL).

### [30] Bicyclo[1.1.1]pentane (BCP) as a Nonclassical Phenyl Bioisostere
* **Authors & Year:** Antonia F. Stepan, Chakrapani Subramanyam, Ivan V. Efremov, Justin K. Ball, ..., and Christopher J. O'Donnell (2012)
* **Title:** *Application of the Bicyclo[1.1.1]pentane Motif as a Nonclassical Phenyl Ring Bioisostere in the Design of a Potent and Orally Active $\gamma$-Secretase Inhibitor*
* **Venue / Identifier:** *Journal of Medicinal Chemistry*, 55(7), 3414–3424 (2012); DOI: `10.1021/jm300094u`
* **Core Contribution:** Validates the strained **bicyclo[1.1.1]pentane (BCP)** cage (~68 kcal/mol ring strain; ultra-short **1.85 Å nonbonded transannular $\text{C}_1\cdots\text{C}_3$ bridgehead distance**; $180^\circ$ collinear exit vectors) as a metabolically stable, soluble *para*-phenyl/alkyne replacement in clinical candidates.
* **Key Limitation Relative to Our Work:** The 1.85 Å nonbonded transannular $\text{C}_1\cdots\text{C}_3$ contact and bridgehead $s$-character hybridization sit in a severe geometric edge regime for short-range radial basis functions and 6-fold low-barrier bridgehead torsions ($\Delta E^\ddagger < 0.5\text{ kcal/mol}$ vs spuriously corrugated MLIP predictions).

### [31] Strain-Release Amination of Propellanes & Bicyclobutanes
* **Authors & Year:** Ryan Gianatassio, Justin M. Lopchuk, Jie Wang, Chung-Mao Pan, Lara R. Malins, Loïc Prieto, ..., Michael R. Collins, Huw M. L. Davies, and Phil S. Baran (2016)
* **Title:** *Strain-release amination*
* **Venue / Identifier:** *Science*, 351(6270), 241–246 (2016); DOI: `10.1126/science.aad6252`
* **Core Contribution:** Unlocks practical synthesis of hetero-substituted **amino-BCPs, amino-bicyclobutanes, and azetidine-bridged cages** from [1.1.1]propellane and bicyclo[1.1.0]butanes, causing an explosion of N-linked strained cage building blocks in post-2016 pharma patents.
* **Key Limitation Relative to Our Work:** Nitrogen lone-pair hyperconjugation into strained $\text{C–C}$ cage $\sigma^*$ orbitals strongly perturbs $\text{C(bridgehead)–N}$ torsional barriers and pyramidalization—an electronic effect poorly captured by classical force fields and under-sampled in SPICE v1.

### [32] Saturated Bioisosteres of Benzene: The *ortho*/*meta* Frontier
* **Authors & Year:** Pavel K. Mykhailiuk (2019)
* **Title:** *Saturated bioisosteres of benzene: where to go next?*
* **Venue / Identifier:** *Organic & Biomolecular Chemistry*, 17(11), 2839–2849 (2019); DOI: `10.1039/C8OB02812E`
* **Core Contribution:** Systematically maps exit-vector distances ($r, d$) and dihedral angles ($\varphi_1, \varphi_2$) across *para*-benzene cages (BCP, bicyclo[2.2.2]octane, cubane) and identifies the grand unsolved challenge of 2019: designing rigid saturated bioisosteres for ***ortho*- and *meta*-disubstituted benzenes**.
* **Key Limitation Relative to Our Work:** Directly explains the temporal blind spot of MACE-OFF23/QMugs/SPICE: almost *all* validated *ortho*- and *meta*-benzene cages (BCHs, 2-oxa-BCHs, BCHeps, aza-BCHs) were invented *after* 2020 and remain largely trapped in recent journal figures and patent PDFs!

### [33] Bicyclo[2.1.1]hexanes (BCHs) & Bicyclo[3.1.1]heptanes (BCHeps) as *ortho*/*meta*-Arene Bioisosteres
* **Authors & Year:** Roman Kleinmans, Tobias Pinkert, Subhabrata Dutta, Constantin G. Daniliuc, Haoming Guo, Stefan Grimme, and Frank Glorius (2022)
* **Title:** *Intermolecular $[2\pi+2\sigma]$-photocycloaddition enabled by triplet energy transfer*
* **Venue / Identifier:** *Nature*, 605, 477–482 (2022); DOI: `10.1038/s41586-022-04636-x` *(paired with Nils Frank, Jeremy Nugent, ..., Pavel K. Mykhailiuk, & Edward A. Anderson, "Synthesis of meta-substituted arene bioisosteres from [3.1.1]propellane", Nature, 611, 721–726, 2022)*
* **Core Contribution:** Establishes **1,2- / 1,4- / 2,4-disubstituted bicyclo[2.1.1]hexanes (BCHs)** via triplet $[2\pi+2\sigma]$ photocycloaddition and **bicyclo[3.1.1]heptanes (BCHeps)** from [3.1.1]propellane as true geometric *ortho*- and *meta*-substituted benzene bioisosteres.
* **Key Limitation Relative to Our Work:** Published in mid/late 2022—completely absent from ChEMBL27/QMugs and virtually absent from SPICE v1! Asymmetric cyclobutane-bridged exit vectors exhibit coupled ring-puckering and steric clash dihedrals ideal for testing MLIP epistemic blind spots.

### [34] 2-Oxabicyclo[2.1.1]hexanes as Water-Soluble *ortho*-Phenyl Bioisosteres
* **Authors & Year:** Aleksandr Denisenko, Pavel Garbuz, Nataliya M. Voloshchuk, Yuliia Holota, Galeb Al-Maali, Petro Borysko, and Pavel K. Mykhailiuk (2023; building on Vadym V. Levterov et al., *Angew. Chem. Int. Ed.* 2020)
* **Title:** *2-Oxabicyclo[2.1.1]hexanes as saturated bioisosteres of the ortho-substituted phenyl ring*
* **Venue / Identifier:** *Nature Chemistry*, 15, 1155–1163 (2023); DOI: `10.1038/s41557-023-01222-0` *(extending Levterov, Panasyuk, Pivnytska & Mykhailiuk, Angew. Chem. Int. Ed., 59(18), 7161–7167, 2020)*
* **Core Contribution:** Replaces a $\text{CH}_2$ bridge of BCH with an ethereal oxygen atom (**2-oxabicyclo[2.1.1]hexane**), lowering $\log P$ by ~1.0–1.5 units, boosting water solubility up to $10\times$, and reproducing *ortho*-phenyl exit vectors in fluxapyroxad/boscalid analogues.
* **Key Limitation Relative to Our Work:** Introduces strong **intramolecular anomeric / $\text{C–O }\sigma^*$ stereoelectronic preferences** onto a strained oxetane-fused bicyclic cage—creating sharp, asymmetric rotational profiles on adjacent amide/aryl substituents where uncalibrated MLIPs systematically soften or misplace minima.

### [35] Cubanes as Validated 3D Benzene Bioisosteres
* **Authors & Year:** Benjamin A. Chalmers, Hui Xing, Sevan Houston, Charlotte Clark, ..., John Tsanaktsidis, G. Paul Savage, and Craig M. Williams (2016)
* **Title:** *Validating Eaton's Hypothesis: Cubane as a Benzene Bioisostere*
* **Venue / Identifier:** *Angewandte Chemie International Edition*, 55(11), 3580–3585 (2016); DOI: `10.1002/anie.201510675`
* **Core Contribution:** Experimentally validates Philip Eaton's hypothesis across 5 bioactive drug/agrochemical scaffolds (letrozole, vorinostat, benzocaine, etc.), proving cubane's 2.72 Å body-diagonal matches benzene's 2.79 Å *para* diameter while retaining or enhancing bioactivity.
* **Key Limitation Relative to Our Work:** Cubane possesses **eight $90^\circ$ $\text{C–C–C}$ bond angles and ~161 kcal/mol strain energy**; standard organic MLIPs trained on $109.5^\circ$–$120^\circ$ hydrocarbons exhibit severe force/hessian errors and bond-softening on substituted cubanes, cuneanes, and homocubanes.

### [36] Fluorinated Motifs, Pentafluorosulfanyl ($\text{SF}_5$), & Strained Oxetane/Spirocyclic Bioisosteres
* **Authors & Year:** Nicholas A. Meanwell (2018)
* **Title:** *Fluorine and Fluorinated Motifs in the Design and Application of Bioisosteres for Drug Design*
* **Venue / Identifier:** *Journal of Medicinal Chemistry*, 61(14), 5822–5880 (2018); DOI: `10.1021/acs.jmedchem.7b01788` *(paired with Meanwell, J. Agric. Food Chem. 2023, 71, 18087–18122 on $\text{SF}_5$ / sulfoximines, and Burkhard, Wuitschik, Rogers-Evans, Müller & Carreira, "Oxetanes as Versatile Elements in Drug Discovery and Synthesis", Angew. Chem. Int. Ed., 49(48), 9052–9067, 2010; Carreira & Fessard, Chem. Rev., 114, 8257–8322, 2014)*
* **Core Contribution:** Codifies tactical bioisosteres including octahedral **pentafluorosulfanyl ($\text{–SF}_5$)** ("super-$\text{CF}_3$" / *tert*-butyl mimetic), difluoromethyl/fluorovinyl amide mimetics, chiral $S(\text{VI})$ **sulfonimidamides**, and **3,3-disubstituted oxetanes / 2-oxa-6-azaspiro[3.3]heptanes** (gem-dimethyl / carbonyl / morpholine bioisosteres).
* **Key Limitation Relative to Our Work:** Notice that $\text{–SF}_5$, $\text{–CF}_3$, oxetanes, and sulfonimidamides contain **only `{H, C, N, O, F, S}`**—100% inside MACE-OFF23's 10-element vocabulary! Yet hypervalent octahedral $\text{S(VI)F}_5$ attached to aryls or strained cages exhibits extreme sterics and 3-center-4-electron hypervalent bonding that causes catastrophic MLIP torsional barrier errors if under-sampled.

---

## Cluster 2E — Foundation MLIP Fine-Tuning, Multi-Head Replay, Softening Errors & Distillation (6 Papers)

### [37] Systematic PES Softening in Universal MLIPs
* **Authors & Year:** Bowen Deng, Yunyeong Choi, Peichen Zhong, Janosh Riebesell, Shashwat Anand, Zhuohan Li, KyuJung Jun, Kristin A. Persson, and Gerbrand Ceder (2025; preprint May 2024)
* **Title:** *Systematic softening in universal machine learning interatomic potentials* *(preprint title: "Overcoming systematic softening in universal machine learning interatomic potentials by fine-tuning")*
* **Venue / Identifier:** *npj Computational Materials*, 11, 1 (2025); DOI: `10.1038/s41524-024-01500-6`; `arXiv:2405.07105`
* **Core Contribution:** Discovers a universal pathology across foundational MLIPs (MACE-MP-0, CHGNet, M3GNet): **systematic Potential Energy Surface (PES) softening**—consistent underprediction of energy barriers, forces, and Hessian curvatures in out-of-distribution / off-equilibrium states because pre-training sets over-sample near-equilibrium minima. Crucially proves that targeted fine-tuning on even a handful of barrier/high-energy configurations eliminates softening!
* **Key Limitation Relative to Our Work:** Demonstrated exclusively on inorganic crystals (phonons, defects, ion migration barriers). **Nobody has yet quantified and cured systematic torsional/strain PES softening in organic foundation MLIPs (MACE-OFF23 / AIMNet2) on modern strained 3D bioisosteres!**

### [38] Data-Efficient Fine-Tuning & Multi-Head Replay for MACE Foundation Models
* **Authors & Year:** Harveen Kaur, Flaviano Della Pia, Ilyes Batatia, Xavier R. Advincula, Benjamin X. Shi, Jinggang Lan, Gábor Csányi, Angelos Michaelides, and Venkat Kapil (2025; preprint May 2024)
* **Title:** *Data-efficient fine-tuning of foundational models for first-principles quality sublimation enthalpies*
* **Venue / Identifier:** *Faraday Discussions*, 256, 120–138 (2025); DOI: `10.1039/D4FD00107A`; `arXiv:2405.20217` *(alongside MACE's official `multi-head replay` protocol in `ACEsuit/mace` by Batatia & Csányi, 2024–2025)*
* **Core Contribution:** Shows that fine-tuning MACE foundation models achieves sub-kJ/mol accuracy on molecular crystals with only tens of structures, while **multi-head replay** (co-training a shared equivariant backbone with task/replay readout heads on FPS-subsampled pre-training replay configurations) prevents **catastrophic forgetting** of general chemical space.
* **Key Limitation Relative to Our Work:** Applied to static crystal polymorphs rather than continuous active-distillation of literature-mined organic bioisosteres, where preserving general SPICE/COMP6/TorsionNet-500 accuracy while curing caged-bioisostere blind spots is essential.

### [39] $\Delta$-Machine Learning ($\Delta$-ML)
* **Authors & Year:** Raghunathan Ramakrishnan, Pavlo O. Dral, Matthias Rupp, and O. Anatole von Lilienfeld (2015)
* **Title:** *Big Data Meets Quantum Chemistry Approximations: The $\Delta$-Machine Learning Approach*
* **Venue / Identifier:** *Journal of Chemical Theory and Computation (JCTC)*, 11(5), 2087–2096 (2015); DOI: `10.1021/acs.jctc.5b00099`; `arXiv:1503.04987`
* **Core Contribution:** Learns smooth residual corrections $\Delta E(\mathbf{R}) = E_{\text{high}}(\mathbf{R}) - E_{\text{base}}(\mathbf{R})$ atop a fast physical baseline (semi-empirical or lower-rung QM/MLIP), slashing data requirements by $10\times$ for chemical accuracy.
* **Key Limitation Relative to Our Work:** Original formulation requires running a slow semi-empirical QM baseline (e.g., PM7 or GFN2-xTB) at every inference step; in contrast, **residual/multi-head fine-tuning or direct student distillation** bakes the $\omega$B97M-D3(BJ)/def2-TZVPPD correction directly into zero-extra-cost inference weights!

### [40] Espaloma & Espaloma-0.3 Differentiable Surrogate Force Fields
* **Authors & Year:** Kenichiro Takaba, Yuanqing Wang, Jeffry Setiadi, Iván Pulido, Michael M. Henry, ..., David L. Mobley, and John D. Chodera (2024; extending Wang et al., *Chem. Sci.* 2022)
* **Title:** *Machine-learned molecular mechanics force fields from large-scale quantum chemical data*
* **Venue / Identifier:** *Chemical Science*, 15(32), 12861–12878 (2024); DOI: `10.1039/D4SC00690A` *(extending Yuanqing Wang et al., "End-to-end differentiable construction of molecular mechanics force fields", Chem. Sci., 13, 12016–12033, 2022)*
* **Core Contribution:** Uses a GNN *offline* once per molecular graph to predict Class I/II MM bond/angle/torsion parameters trained end-to-end on SPICE energies and forces (`espaloma-0.3`), achieving classical OpenMM MD speeds for protein–ligand free-energy calculations.
* **Key Limitation Relative to Our Work:** Restricted to fixed harmonic/Fourier MM potential functional forms—cannot capture continuous environment-dependent bond polarization, transannular through-space bridgehead repulsion in BCP/BCH cages, or coupled non-harmonic strain-torsion cross-terms.

### [41] Accelerating Equivariant Molecular GNNs via Knowledge Distillation
* **Authors & Year:** Filip Ekström Kelvinius, Dimitar Georgiev, Artur Petrov Toshev, and Johannes Gasteiger (2023)
* **Title:** *Accelerating Molecular Graph Neural Networks via Knowledge Distillation*
* **Venue / Identifier:** *Advances in Neural Information Processing Systems (NeurIPS)*, 36 (2023); `arXiv:2306.14818`
* **Core Contribution:** Formulates node-feature, vector/spherical-tensor representation, and Hessian/force-matching **knowledge distillation** from heavy equivariant teachers into fast compact student GNNs (`node2node`, `vec2vec`, force-augmented trajectories), recovering up to 96.7% of the teacher–student energy gap and 62.5% of the force gap at $3\times$–$8\times$ faster throughput.
* **Key Limitation Relative to Our Work:** Distills a *flawed* teacher's existing distribution on static OC20/MD17 datasets: if the teacher (e.g., MACE-OFF23-large) has epistemic blind spots and PES softening on modern bioisosteres, pure teacher-student distillation faithfully copies the teacher's blind spots unless combined with **targeted QM active-learning oracle calls**!

### [42] EGraFFBench & OOD Audits of Universal Force Fields
* **Authors & Year:** Vaibhav Bihani, Sajid Mannan, Utkarsh Pratiush, Tao Du, Zhimin Chen, Santiago Miret, Matthieu Micoulaut, Morten M. Smedskjaer, Sayan Ranu, and N. M. Anoop Krishnan (2024)
* **Title:** *EGraFFBench: evaluation of equivariant graph neural network force fields for atomistic simulations*
* **Venue / Identifier:** *Digital Discovery*, 3(4), 759–768 (2024); DOI: `10.1039/D4DD00027G` *(complemented by Focassio, Freitas & Schleder, "Performance Assessment of Universal Machine Learning Interatomic Potentials: Challenges and Directions for Materials’ Surfaces", ACS Appl. Mater. Interfaces / arXiv:2403.04217, 2024)*
* **Core Contribution:** Proves across NequIP, Allegro, MACE, BOTNet, Equiformer, and TorchMD-Net that **low static force MAE on in-distribution test splits fails to guarantee MD trajectory stability or OOD accuracy**, introducing trajectory stability horizon, radial distribution fidelity, and OOD temperature/structure stress tests.
* **Key Limitation Relative to Our Work:** Benchmarks inorganic glasses/electrolytes and surfaces rather than medicinal-chemistry conformational/torsional PES smoothness, derivative continuity, and cage ring-integrity under finite-temperature MD.

---

## Prior-Art Novelty Audit: Has Anyone Combined OCSR Visual Mining $\to$ MLIP Blind-Spot Discovery $\to$ Active QM Distillation?

**Verified Conclusion: NO.** Exhaustive cross-domain searches spanning OCSR tools (*MolScribe*, Qian et al., *JCIM* 2023; *DECIMER.ai*, Rajan et al., *Nat. Commun.* 2023), MLIP active learning (*ANI-1x*, *DP-GEN*, *LCMD*, *UDD-AL*), and MLIP distillation/fine-tuning (*MACE-OFF23*, *Kaur et al.*, *Ekström Kelvinius et al.*) confirm **zero prior published or arXiv works** connecting these three stages:

1. **OCSR literature/patent pipelines** stop at 2D SMILES/graph extraction or QSAR bioactivity table curation—they have never been used as an **epistemic chemical-space radar** to expose where universal 3D quantum force fields fail on newly invented synthetic motifs.
2. **MLIP active-learning pipelines** (*Smith 2018*, *Zaverkin 2022*, *Kulichenko 2023*) sample only inside static legacy SMILES lists (GDB, pre-2021 ChEMBL, PubChem) or single-molecule MD trajectories—making them **structurally blind** to the 2020–2026 wave of saturated 3D bioisosteres (`BCH`, `2-oxa-BCH`, `BCHep`, `aza-BCP`, `cubane`, `SF5-cage`, `oxetane-spirocycle`) depicted visually in synthetic/medchem papers and patents years before database curation.
3. **MLIP knowledge distillation** (*Ekström Kelvinius 2023*) compresses teacher models on static benchmarks without **hybrid teacher + selective DFT oracle correction**, whereas our pipeline uses uncertainty-gated Psi4 $\omega$B97M-D3(BJ)/def2-TZVPPD calculations *only* where the teacher itself is blind, while distilling the teacher's accurate predictions everywhere else via multi-head replay!

---

## Gap & Creativity Synthesis Matrix for `002-mlip-blindspot-distill`

| Dimension | Standard Prior Art | Failure Mode on Modern MedChem | Concrete Creative Upgrade for `002-mlip-blindspot-distill` |
| :--- | :--- | :--- | :--- |
| **1. Chemical Space Source** | Static pre-2021 ChEMBL27 (QMugs) or random PubChem $\le 50$ atoms (SPICE v1) | Misses post-2020 3D bioisosteres (*ortho*/*meta* BCHs, 2-oxa-BCHs, BCHeps, cubanes, $\text{SF}_5$-cages) locked in journal schemes & patent figures | **Visual Literature/Patent OCSR Radar** (`MolScribe`/`DECIMER` + RDKit stereo/valence + **MACE-OFF23 10-element & neutral-singlet gate**: strip salts, canonicalize neutral protomers, route `B`/`Si` or ions separately). |
| **2. Blind-Spot Detection** | Expensive $8\times$ deep ensembles trained from scratch (*Smith 2018*, *DP-GEN*) | Prohibitive to train 8 MACE-OFF23-large models; single-signal latent density can miss subtle electronic barriers | **Zero-Retrain Tri-Signal Epistemic Detector**: combine (a) **cross-architecture disagreement** (`MACE-OFF23` vs `AIMNet2` relative conformer/barrier $\Delta E$ & force cosine/magnitude), (b) **MACE per-atom scalar latent Mahalanobis/GMM OOD score** ([26]), and (c) **last-layer gradient LCMD batch selection** ([24]). |
| **3. Physical Failure Diagnosis** | Aggregate static energy/force MAE on random equilibrium conformers | Masks **systematic PES softening** ([37]) at torsional barrier tops ($\Delta\text{BH}$), spurious barrier corrugation, and cage bridgehead angle distortion | **Bioisostere Torsional Softening & Strain Diagnostic Suite**: relaxed 1D torsion scans ($15^\circ$ grid) on exit-vector bonds + cage bridgehead distance/angle distortion ($\text{C}_1\cdots\text{C}_3$) + harmonic/finite-diff curvature softening Slope $\alpha < 1$ ([37]). |
| **4. Quantum Oracle Alignment** | Mixed DFT levels (`def2-SVP` in QMugs, `def2-TZVPP` in AIMNet2, `def2-TZVPD`/VV10 in OMol25) | Direct single-head fine-tuning on mismatched basis/functional corrupts MACE-OFF23's energy scale and forces | **Strict Psi4 $\omega$B97M-D3(BJ)/def2-TZVPPD Oracle** on LCMD-selected high-uncertainty torsion/off-equilibrium frames, paired with **hybrid Teacher-Label + Selective-QM Active Distillation**. |
| **5. Model Update & Speed** | Naive full fine-tuning (catastrophic forgetting) OR pure teacher distillation (copies teacher's blind spots) | Forgets general SPICE/COMP6/TorsionNet-500 accuracy OR remains too slow (`MACE-OFF23-large`) for high-throughput MD | **Replay-Protected Multi-Head Fine-Tuning + Calibrated Shallow-Ensemble Student Distillation**: cure `MACE-OFF23` via SPICE/TorsionNet replay ([38]), then distill into a fast `MACE-OFF-small` student equipped with a **Kellner–Ceriotti shallow-ensemble readout head** ([27]) for built-in $1\times$-cost calibrated UQ & MD stability ([42]). |
