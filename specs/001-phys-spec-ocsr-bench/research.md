# Exhaustive Verified Literature Review: Application 1 + Frontier Benchmark (`001-phys-spec-ocsr-bench`)
## Thermodynamically & Spectroscopically Informed OCSR + Multimodal AI4Science Chemistry Benchmark

**Total Verified Primary Entries:** 42 peer-reviewed / arXiv papers across 5 thematic clusters (plus 12 directly linked companion papers noted inline, all verified via `search_web`).

---

## Cluster 1A — Optical Chemical Structure Recognition (OCSR), Stereochemistry & Real-Document Domain Gap (Papers 1–10)

### 1. DECIMER.ai
- **Authors:** Kohulan Rajan, Henning Otto Brinkhaus, M. Isabel Agea, Achim Zielesny, and Christoph Steinbeck
- **Year:** 2023
- **Title:** *DECIMER.ai: an open platform for automated optical chemical structure identification, segmentation and recognition in scientific publications*
- **Venue / Identifier:** *Nature Communications*, 14, 5045 (`DOI: 10.1038/s41467-023-40782-0`) [building on Rajan, Zielesny & Steinbeck, *DECIMER 1.0: deep learning for chemical image recognition using transformers*, *J. Cheminform.* 2021, `DOI: 10.1186/s13321-021-00538-8`; extended to ensemble curation in Rajan et al., *MARCUS: molecular annotation and recognition for curating unravelled structures*, *Digital Discovery* 2025]
- **Core Contribution:** End-to-end open platform combining Mask R-CNN page segmentation with an EfficientNetV2 + Transformer image-to-SELFIES/SMILES decoder trained on >400M synthetic depictions plus hand-drawn augmentations.
- **Key Limitation Relative to Our Work:** Purely 2D pixel-to-sequence translation with zero 3D geometric, thermodynamic, or spectroscopic feedback; exhibits steep accuracy drops on tetrahedral (`@`/`@@`) and bridged-ring stereochemistry in degraded journal scans, which even MARCUS (2025) can only mitigate via human-in-the-loop multi-OCSR voting rather than physical/analytical verification.

### 2. MolScribe
- **Authors:** Yujie Qian, Jiang Guo, Zhengkai Tu, Zhening Li, Connor W. Coley, and Regina Barzilay
- **Year:** 2023
- **Title:** *MolScribe: Robust Molecular Structure Recognition with Image-to-Graph Generation*
- **Venue / Identifier:** *Journal of Chemical Information and Modeling*, 63(7), 1925–1934 (`DOI: 10.1021/acs.jcim.2c01480`)
- **Core Contribution:** Formulates OCSR as conditional image-to-graph generation (Swin Transformer encoder + autoregressive atom/coordinate/bond token decoder) with rule-based superatom expansion and 2D coordinate-based stereochemical assignment.
- **Key Limitation Relative to Our Work:** Assigns chiral centers solely from 2D predicted vertex coordinates and local wedge/dash bond labels; when low-DPI scans, bond crossings, or bridged/polycyclic perspective projections distort 2D wedge geometry, MolScribe silently emits diastereomerically inverted or thermodynamically impossible 3D ring junctions (empirically dropping from 98% skeleton accuracy to 83% single-stereocenter accuracy on real Supporting Information PDFs in NMRexp).

### 3. MolGrapher
- **Authors:** Lucas Morin, Martin Danelljan, Maria Isabel Agea, Ahmed Nassar, Valery Weber, Ingmar Meijer, Peter Staar, and Fisher Yu
- **Year:** 2023
- **Title:** *MolGrapher: Graph-based Visual Recognition of Chemical Structures*
- **Venue / Identifier:** *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 19552–19561 (`arXiv:2308.12234`)
- **Core Contribution:** Two-stage keypoint detection + GNN node/edge classification architecture that explicitly localizes atoms and bonds in the image plane before constructing the molecular graph.
- **Key Limitation Relative to Our Work:** Keypoint localization errors on crowded natural-product scaffolds, bridged bicycles, or overlapping R-group/atom labels propagate irreversibly into graph topology and wedge orientation without any post-hoc 3D conformer strain or NMR/HRMS consistency check.

### 4. SwinOCSR
- **Authors:** Zhanpeng Xu, Jianhua Li, Zhaopeng Yang, Shilong Li, and Honglin Li
- **Year:** 2022
- **Title:** *SwinOCSR: end-to-end optical chemical structure recognition using a Swin Transformer*
- **Venue / Identifier:** *Journal of Cheminformatics*, 14, 41 (`DOI: 10.1186/s13321-022-00624-5`)
- **Core Contribution:** Applies shifted-window hierarchical vision transformers with focal loss to map molecular images to DeepSMILES strings, mitigating class imbalance in rare tokens.
- **Key Limitation Relative to Our Work:** Trained predominantly on clean synthetic renders without stereochemical supervision or real-world literature noise; omits chiral/cis-trans fidelity on complex stereoisomers and lacks any physical or analytical post-verification.

### 5. ChemPix
- **Authors:** Hayley Weir, Keiran Thompson, Amelia Woodward, Benjamin Choi, Augustin Braun, and Todd J. Martínez
- **Year:** 2021
- **Title:** *ChemPix: automated recognition of hand-drawn hydrocarbon structures using deep learning*
- **Venue / Identifier:** *Chemical Science*, 12(31), 10622–10633 (`DOI: 10.1039/D1SC02957F`)
- **Core Contribution:** Pioneered synthetic-to-real image degradation and ensemble voting with calibrated abstention for recognizing noisy smartphone photographs of hand-drawn hydrocarbon structures.
- **Key Limitation Relative to Our Work:** Restricted to simple achiral/unfunctionalized hydrocarbons; relies purely on neural ensemble variance for uncertainty estimation rather than grounded physical (3D strain) or experimental (NMR/HRMS) falsification.

### 6. OpenChemIE (and RxnScribe)
- **Authors:** Vincent Fan, Yujie Qian, Alex Wang, Amber Wang, Connor W. Coley, and Regina Barzilay
- **Year:** 2024
- **Title:** *OpenChemIE: An Information Extraction Toolkit for Chemistry Literature*
- **Venue / Identifier:** *Journal of Chemical Information and Modeling*, 64(14), 5521–5534 (`DOI: 10.1021/acs.jcim.4c00572`) [incorporating Qian et al., *RxnScribe: A Sequence Generation Model for Reaction Diagram Parsing*, *JCIM* 2023, 63(13), 4030–4041, `DOI: 10.1021/acs.jcim.3c00439`]
- **Core Contribution:** Integrates reaction scheme parsing (RxnScribe), molecule recognition (MolScribe), coreference resolution ( compound labels like **3a**), and substrate-scope table extraction across PDF figures and text.
- **Key Limitation Relative to Our Work:** Links compound numbers (**3a**) across reaction schemes and text, yet stops short of extracting the corresponding Supporting Information experimental characterization block ($^1\text{H}/^{13}\text{C}$ NMR, HRMS) to verify or disambiguate the OCSR-extracted stereostructure.

### 7. MolNexTR
- **Authors:** Yufan Chen, Ching Ting Leung, Yong Huang, Jianwei Sun, Hao Chen, and Hanyu Gao
- **Year:** 2024
- **Title:** *MolNexTR: a generalized deep learning model for molecular image recognition*
- **Venue / Identifier:** *Journal of Cheminformatics*, 16, 141 (`DOI: 10.1186/s13321-024-00926-w`; `arXiv:2403.03691`)
- **Core Contribution:** Dual-backbone ConvNeXt + Vision Transformer encoder paired with image-to-graph decoding, enhanced data augmentations, and improved stereochemical rule algorithms for noisy and hand-drawn depictions.
- **Key Limitation Relative to Our Work:** Still operates strictly in 2D image space; when visual cues are genuinely ambiguous (faint dashed bonds, perspective Haworth/chair drawings, intersecting bridged bonds), 2D geometric heuristics cannot adjudicate between competing diastereomers without 3D energetics or spectroscopic evidence.

### 8. AtomLenz
- **Authors:** Martijn Oldenhof, Edward De Brouwer, Ádám Arany, and Yves Moreau
- **Year:** 2024
- **Title:** *Atom-Level Optical Chemical Structure Recognition with Limited Supervision*
- **Venue / Identifier:** *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 18082–18091 (`arXiv:2404.01743`)
- **Core Contribution:** Introduces probabilistic knowledge-transfer training (`ProbKT*`) enabling Faster R-CNN atom/bond/charge/stereocenter localization using only SMILES-level supervision without pixel-level bounding-box labels.
- **Key Limitation Relative to Our Work:** Addresses annotation scarcity on sparse domains (e.g., hand-drawn sketches) purely visually; produces uncalibrated 2D topological/chiral graphs with no thermodynamic feasibility filter or cross-modal spectral grounding.

### 9. MolSight (and MolParser)
- **Authors:** Wenrui Zhang, Xinggang Wang, Bin Feng, and Wenyu Liu
- **Year:** 2025 (AAAI 2026)
- **Title:** *MolSight: Optical Chemical Structure Recognition with SMILES Pretraining, Multi-Granularity Learning and Reinforcement Learning*
- **Venue / Identifier:** *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2026)* (`arXiv:2511.17300`) [building on Xi Fang et al., *MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild*, *ICCV 2025*, which introduced `MolParser-7M` and E-SMILES]
- **Core Contribution:** Directly targets OCSR's stereochemistry bottleneck by constructing **Stereo-200K** (200k confusing stereoisomer images) and training a 3-stage VLM with Group Relative Policy Optimization (GRPO) RL rewards for chemical semantic/stereochemical exactness.
- **Key Limitation Relative to Our Work:** Proves that stereochemistry is the #1 unsolved frontier in OCSR, yet tackles it purely via larger synthetic/curated 2D training sets and RL over image-only inputs. In real organic synthesis papers, compressed journal figures often make wedge/dash orientation *information-theoretically underdetermined from pixels alone*—resolvable only by consulting 3D Boltzmann strain and the compound's accompanying NMR/HRMS data.

### 10. General $E(2)$-Equivariant Steerable CNNs
- **Authors:** Maurice Weiler and Gabriele Cesa
- **Year:** 2019
- **Title:** *General $E(2)$-Equivariant Steerable CNNs*
- **Venue / Identifier:** *Advances in Neural Information Processing Systems (NeurIPS)*, 32 (`arXiv:1911.08251`)
- **Core Contribution:** Establishes the unified group-representation theory (`e2cnn`) for planar translation-, rotation-, and reflection-equivariant convolutions via irreducible representations of $E(2)$ and its subgroups ($SO(2), O(2), C_n, D_n$).
- **Key Limitation Relative to Our Work:** Crucial geometric subtlety for OCSR: tetrahedral stereochemical depictions are **$SO(2)$-invariant under planar rotation** (rotating a 2D drawing in-plane preserves $R/S$), **NOT $O(2)$-reflection-invariant** (mirroring an image in 2D inverts all stereocenters into the enantiomer!). Naive reflection augmentation or $O(2)$-invariant pooling destroys chirality; pairing $SO(2)$-aware visual perception with 3D $SE(3)$/chiral-sensitive physics + NMR verification resolves planar orientation ambiguity cleanly.

---

## Cluster 1B — Computational NMR / MS Structure Elucidation & Diastereomer Assignment (Papers 11–21)

### 11. DP4
- **Authors:** Steven G. Smith and Jonathan M. Goodman
- **Year:** 2010
- **Title:** *Assigning Stereochemistry to Single Diastereoisomers by GIAO NMR Calculation: The DP4 Probability*
- **Venue / Identifier:** *Journal of the American Chemical Society*, 132(37), 12946–12959 (`DOI: 10.1021/ja105035r`)
- **Core Contribution:** Landmark Bayesian framework assigning relative stereochemistry among $2^{n-1}$ candidate diastereomers when *only a single experimental $^1\text{H}/^{13}\text{C}$ NMR spectrum* is available, using Student-$t$ error distributions of Boltzmann-weighted DFT GIAO chemical shifts.
- **Key Limitation Relative to Our Work:** Requires minutes-to-hours of B3LYP/6-31G(d,p)//MMFF DFT per diastereomer and manual peak-to-atom assignment, and has never been connected to OCSR candidate/stereoisomer hypothesis reranking.

### 12. DP4+
- **Authors:** Nicolás Grimblat, María M. Zanardi, and Ariel M. Sarotti
- **Year:** 2015
- **Title:** *Beyond DP4: an Improved Probability for the Stereochemical Assignment of Isomeric Compounds using Quantum Chemical Calculations of NMR Shifts*
- **Venue / Identifier:** *The Journal of Organic Chemistry*, 80(24), 12526–12534 (`DOI: 10.1021/acs.joc.5b02396`)
- **Core Contribution:** Upgrades DP4 by combining scaled and unscaled chemical shift residuals with PCM solvent modeling (mPW1PW91/6-31+G**//B3LYP/6-31G*) and hybridization-specific ($sp^2$ vs. $sp^3$) error distributions.
- **Key Limitation Relative to Our Work:** Even higher quantum-chemical cost per conformer ensemble makes full DFT DP4+ prohibitive for high-throughput literature mining unless replaced or pre-screened by millisecond 3D MLIP conformer ensembles + 3D GNN NMR shift predictors.

### 13. ML-J-DP4 (and J-DP4)
- **Authors:** Yi-Hsuan Tsai, Milagros Amichetti, María Marta Zanardi, Rafael Grimson, Antonio Hernández Daranas, and Ariel M. Sarotti
- **Year:** 2022
- **Title:** *ML-J-DP4: An Integrated Quantum Mechanics-Machine Learning Approach for Ultrafast NMR Structural Elucidation*
- **Venue / Identifier:** *Organic Letters*, 24(41), 7487–7491 (`DOI: 10.1021/acs.orglett.2c02782`) [extending Grimblat et al., *Combining the Power of $J$ Coupling and DP4 Analysis on Stereochemical Assignment: The J-DP4 Methods*, *Org. Lett.* 2019, 21(11), 4003–4007]
- **Core Contribution:** Integrates $^3J_{\text{HH}}$ vicinal coupling constants (Karplus/dihedral geometric constraints) with ML-accelerated isotropic shieldings over fast low-level geometries, achieving ~100× speedup over DFT DP4+ while sharpening diastereomer discrimination.
- **Key Limitation Relative to Our Work:** Designed as a standalone interactive tool for chemists elucidating an unknown natural product—never formulated to adjudicate visual OCSR ambiguity or screen OCSR beam/epimer hypotheses at scale.

### 14. DP4-AI
- **Authors:** Alexander Howarth, Kristaps Ermanis, and Jonathan M. Goodman
- **Year:** 2020
- **Title:** *DP4-AI automated NMR data analysis: straight from spectrometer to structure*
- **Venue / Identifier:** *Chemical Science*, 11(17), 4351–4359 (`DOI: 10.1039/D0SC00442A`)
- **Core Contribution:** Fully automates raw FID/peak-picked NMR processing and bipartite peak-to-atom assignment (Hungarian/combinatorial matching) for automated diastereomer probability scoring without human atom-number mapping.
- **Key Limitation Relative to Our Work:** Crucially demonstrates *unassigned* peak-list matching (which directly mirrors unassigned SI text peak tables!), but relies on expensive DFT GIAO calculations and assumes the candidate set is manually supplied rather than generated from uncertain OCSR stereocenters.

### 15. CASCADE
- **Authors:** Yanfei Guan, Sree Souris S. V. Sowndarya, Luis C. Gallegos, Peter C. St. John, and Robert S. Paton
- **Year:** 2021
- **Title:** *Real-time prediction of $^1\text{H}$ and $^{13}\text{C}$ chemical shifts with DFT accuracy using a 3D graph neural network*
- **Venue / Identifier:** *Chemical Science*, 12(36), 12012–12026 (`DOI: 10.1039/D1SC03343C`)
- **Core Contribution:** 3D message-passing GNN operating on conformer ensembles (RBF interatomic distances) trained on DFT + experimental shifts (`Exp5K`), achieving sub-second $^1\text{H}/^{13}\text{C}$ chemical shift predictions with DFT-level accuracy (~1.26 ppm $^{13}\text{C}$ MAE) capable of resolving regio- and diastereoisomers.
- **Key Limitation Relative to Our Work:** Provides the exact millisecond 3D-stereosensitive forward NMR engine needed to replace DFT GIAO in high-throughput verification, yet has never been coupled to OCSR, MLIP Boltzmann weighting (MACE-OFF), or unassigned SI peak-table likelihood scoring.

### 16. SGNN
- **Authors:** Eric Jonas and Stefan Kuhn
- **Year:** 2019
- **Title:** *Rapid prediction of NMR spectral properties with quantified uncertainty*
- **Venue / Identifier:** *Journal of Cheminformatics*, 11, 50 (`DOI: 10.1186/s13321-019-0374-3`)
- **Core Contribution:** Fast 2D molecular graph neural network predicting $^1\text{H}$ and $^{13}\text{C}$ NMR shifts with calibrated per-atom uncertainty estimates trained on NMRShiftDB2.
- **Key Limitation Relative to Our Work:** Uses 2D topological connectivity (with at most rudimentary stereo flags) without 3D through-space distances or Boltzmann conformer ensembles—making 2D GNNs unable to capture through-space $\gamma$-gauche / anisotropic shielding differences that distinguish polycyclic diastereomers!

### 17. IMPRESSION
- **Authors:** Will Gerrard, Lars A. Bratholm, Martin J. Packer, Adrian J. Mulholland, David R. Glowacki, and Craig P. Butts
- **Year:** 2020
- **Title:** *IMPRESSION – prediction of NMR parameters for 3-dimensional chemical structures using machine learning with near quantum chemical accuracy*
- **Venue / Identifier:** *Chemical Science*, 11(2), 508–515 (`DOI: 10.1039/C9SC03854J`)
- **Core Contribution:** Kernel-ridge/3D descriptor ML framework predicting *both* $^{13}\text{C}/^1\text{H}$ chemical shifts AND $^1J_{\text{CH}}$ scalar coupling constants directly from 3D coordinates at near-DFT accuracy in milliseconds, flagging stereochemical misassignments.
- **Key Limitation Relative to Our Work:** Limited element coverage (C, H, N, O, F) and single-conformer training regime; never integrated with conformer Boltzmann weighting or multimodal visual-spectroscopic OCSR reranking.

### 18. Multimodal Spectroscopic Dataset & Transformer Structure Elucidation
- **Authors:** Marvin Alberts, Oliver Schilter, Federico Zipoli, Nina Hartrampf, and Teodoro Laino
- **Year:** 2024
- **Title:** *Unraveling Molecular Structure: A Multimodal Spectroscopic Dataset for Chemistry*
- **Venue / Identifier:** *Advances in Neural Information Processing Systems (NeurIPS 2024, Datasets and Benchmarks Track)* (`arXiv:2407.17492`) [paired with Alberts, Laino & Vaucher, *Leveraging infrared spectroscopy for automated structure elucidation*, *Commun. Chem.* 2024, 7, 268; and Alberts, Zipoli & Laino, *Digital Discovery* 2025]
- **Core Contribution:** Releases ~790k molecules with simulated $^1\text{H}$ NMR, $^{13}\text{C}$ NMR, HSQC, IR, and MS spectra and benchmarks multimodal sequence-to-SMILES transformers for *de novo* structure elucidation from spectra alone.
- **Key Limitation Relative to Our Work:** Relies on simulated spectra of patent-like small molecules (<35 heavy atoms) and solves unconstrained *de novo* SMILES generation (where accuracy collapses on complex natural products/polycycles), ignoring the rich visual scaffold anchor provided by molecular depictions in papers.

### 19. MassSpecGym
- **Authors:** Roman Bushuiev, Anton Bushuiev, Niek F. de Jonge, Andrew Young, Fabian Kretschmer, Raman Samusevich, ..., Josef Sivic, and Tomáš Pluskal
- **Year:** 2024
- **Title:** *MassSpecGym: A benchmark for the discovery and identification of molecules*
- **Venue / Identifier:** *Advances in Neural Information Processing Systems (NeurIPS 2024, Datasets and Benchmarks Track)* (`arXiv:2410.23326`)
- **Core Contribution:** Standardized, scaffold-split benchmark of 231k experimental MS/MS spectra across *de novo* generation, candidate retrieval, and spectrum simulation, exposing severe data leakage in prior random splits and <5% *de novo* top-1 exactness on novel scaffolds.
- **Key Limitation Relative to Our Work:** Evaluates MS/MS in isolation (which is fundamentally blind to chirality and most diastereomers, and ambiguous across regioisomers), highlighting why MS/HRMS is ideal for exact-formula/adduct/substructure pruning while 3D NMR + energetics must resolve stereochemistry.

### 20. DiffMS
- **Authors:** Montgomery Bohde, Mrunali Manjrekar, Runzhong Wang, Shuiwang Ji, and Connor W. Coley
- **Year:** 2025
- **Title:** *DiffMS: Diffusion Generation of Molecules Conditioned on Mass Spectra*
- **Venue / Identifier:** *Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)* (`arXiv:2502.09571`)
- **Core Contribution:** Formula-constrained discrete graph diffusion decoder pretrained on massive fingerprint-structure pairs and conditioned on a MIST mass-spectrum encoder, achieving SOTA *de novo* generation on MassSpecGym.
- **Key Limitation Relative to Our Work:** Demonstrates the power of **hard exact-formula constraints** (derived from HRMS) during graph decoding, yet produces 2D achiral graphs because MS/MS alone lacks 3D stereochemical observability and visual scheme context.

### 21. DeepSAT (and SMART 2.0)
- **Authors:** Hyun Woo Kim, Chen Zhang, Raphael Reher, Mingxun Wang, ..., Pieter C. Dorrestein, Garrison W. Cottrell, and William H. Gerwick
- **Year:** 2023
- **Title:** *DeepSAT: Learning Molecular Structures from Nuclear Magnetic Resonance Data*
- **Venue / Identifier:** *Journal of Cheminformatics*, 15, 71 (`DOI: 10.1186/s13321-023-00738-4`) [building on Reher et al., *A Convolutional Neural Network-Based Approach for the Rapid Annotation of Molecularly Diverse Natural Products* (SMART 2.0), *JACS* 2020, 142(9), 4114–4120]
- **Core Contribution:** Uses CNN/multi-task learning over $^{1}\text{H}$–$^{13}\text{C}$ HSQC 2D NMR spectra to retrieve scaffolds, classify natural product classes, and predict molecular fingerprints.
- **Key Limitation Relative to Our Work:** Requires full 2D HSQC spectral images/peak coordinates (rarely tabulated in standard synthetic chemistry SIs compared to 1D $^1\text{H}$/$^{13}\text{C}$ text lists) and performs database retrieval rather than exact stereoisomer verification of OCSR outputs.

---

## Cluster 1C — 3D Steric Strain, Geometry Sanity Checks & Silent Chemical Errors (Papers 22–28)

### 22. PoseBusters
- **Authors:** Martin Buttenschoen, Garrett M. Morris, and Charlotte M. Deane
- **Year:** 2024
- **Title:** *PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences*
- **Venue / Identifier:** *Chemical Science*, 15(9), 3130–3139 (`DOI: 10.1039/D3SC04185A`)
- **Core Contribution:** Introduces an 18-check physical validity suite (bond lengths/angles, planarity, internal steric clash, tetrahedral chirality, double-bond stereochemistry, energy ratio) showing that top deep-learning docking models (e.g., DiffDock, EquiBind) frequently output physically impossible geometries despite strong RMSD scores.
- **Key Limitation Relative to Our Work:** Evaluates 3D protein-ligand docking poses given a *known* ground-truth 2D ligand graph; never applied to detect **2D-to-3D OCSR hallucinations** where a valid-valence 2D SMILES produced by OCSR (e.g., inverted bridged ring junction or impossible *trans*-fused small ring) fails 3D conformer embedding or incurs catastrophic internal strain!

### 23. GenBench3D
- **Authors:** Benoit Baillif, Jason Cole, Patrick McCabe, and Andreas Bender
- **Year:** 2024 (JCIM 2025)
- **Title:** *Benchmarking structure-based three-dimensional molecular generative models using GenBench3D: ligand conformation quality matters*
- **Venue / Identifier:** `arXiv:2407.04424` / Peer-reviewed expansion in *Journal of Chemical Information and Modeling*, 2025, 65(19), 8006–8021 (`DOI: 10.1021/acs.jcim.5c01020`)
- **Core Contribution:** Quantifies 3D torsion-angle distributions (CSD Mogul $z$-scores) and quantum/force-field strain energy ($\Delta E_{\text{strain}}$) of AI-generated molecules, showing high 2D RDKit validity masks severe 3D conformational strain.
- **Key Limitation Relative to Our Work:** Benchmarks 3D *generative* models for drug design; overlooks that *perception* models (OCSR and VLMs) suffer an analogous silent failure mode—passing RDKit 2D sanitization while encoding severely strained or stereochemically inverted 3D topologies.

### 24. Auto3D (and MACE-OFF Transferable MLIPs)
- **Authors:** Zhen Liu, Tetiana Zubatiuk, Adrian Roitberg, and Olexandr Isayev
- **Year:** 2022
- **Title:** *Auto3D: Automatic Generation of the Low-Energy 3D Structures with ANI Neural Network Potentials*
- **Venue / Identifier:** *Journal of Chemical Information and Modeling*, 62(22), 5373–5382 (`DOI: 10.1021/acs.jcim.2c00817`) [modernized for broader organic chemistry by Kovács et al., *MACE-OFF: Short-Range Transferable Machine Learning Force Fields for Organic Molecules*, *JACS* 2025, `DOI: 10.1021/jacs.4c07099`]
- **Core Contribution:** Automated pipeline combining RDKit conformer sampling with neural network interatomic potentials (ANI-2x / AIMNet; extensible to MACE-OFF23) to find global minimum-energy 3D conformers, tautomer rankings, and relative stereoisomer energies within <0.5 kcal/mol of DFT in seconds.
- **Key Limitation Relative to Our Work:** Assumes the input constitutional/stereochemical SMILES is intended; never repurposed as a **thermodynamic Bayesian prior over OCSR candidate stereoisomers** to penalize strained diastereomer misreads while remaining fast enough for literature-scale extraction.

### 25. ETKDGv3
- **Authors:** Shuzhe Wang, Jagna Witek, Gregory A. Landrum, and Sereina Riniker
- **Year:** 2020
- **Title:** *Improving Conformer Generation for Small Rings and Macrocycles Based on Distance Geometry and Experimental Torsional-Angle Preferences*
- **Venue / Identifier:** *Journal of Chemical Information and Modeling*, 60(4), 2044–2058 (`DOI: 10.1021/acs.jcim.0c00025`)
- **Core Contribution:** Incorporates CSD-derived small-ring and macrocycle torsional preferences plus distance-geometry chirality/planarity bounds into RDKit's conformer generator.
- **Key Limitation Relative to Our Work:** Distance-geometry embedding convergence failure rates and bounds-violation residuals are treated merely as conformer-generation errors rather than exploited as **fast diagnostic signals of impossible OCSR stereochemistry** (e.g., misread bridged wedges).

### 26. Anti-Bredt Olefins & Continuous Strain Energy
- **Authors:** Luca McDermott, Zach G. Walters, Sarah A. French, Allison M. Clark, Jiaming Ding, Andrew V. Kelleghan, K. N. Houk, and Neil K. Garg
- **Year:** 2024
- **Title:** *A solution to the anti-Bredt olefin synthesis problem*
- **Venue / Identifier:** *Science*, 386(6721), eadq3519 (`DOI: 10.1126/science.adq3519`)
- **Core Contribution:** Overturns a 100-year binary textbook heuristic ("Bredt's rule") by generating and trapping anti-Bredt olefins (ABOs) with DFT-computed olefin strain energies (OSE) of ~30–50+ kcal/mol, proving that **continuous 3D strain energy and pyramidalization/twist energetics**—not brittle binary SMARTS/topological rules—govern what can exist as an isolable product vs. a fleeting unisolable intermediate.
- **Key Limitation Relative to Our Work:** Purely synthetic/computational mechanistic study, yet provides a *profound physical design principle* for `001-phys-spec-ocsr-bench`: binary rule filters fail in both directions, whereas continuous MLIP strain energy ($\Delta E_{\text{strain}}$ relative to stereoisomer floor, calibrated for isolable room-temperature characterization in NMR tubes vs. transient $>25\text{–}30\text{ kcal/mol}$ species) physically separates real isolable compounds from OCSR hallucinations without hard-coded dogma!

### 27. CSD Mogul Geometry Validation
- **Authors:** Ian J. Bruno, Jason C. Cole, Magnus Kessler, Jie Luo, W. D. Sam Motherwell, Lucy H. Purkis, Barry R. Smith, Robin Taylor, Richard I. Cooper, Stephanie E. Harris, and A. Guy Orpen
- **Year:** 2004
- **Title:** *Retrieval of Crystallographically-Derived Molecular Geometry Information*
- **Venue / Identifier:** *Journal of Chemical Information and Computer Sciences*, 44(6), 2133–2144 (`DOI: 10.1021/ci049780b`)
- **Core Contribution:** Establishes knowledge-based validation of bond lengths, valence angles, and ring/acyclic torsion angles against experimental crystal structures in the Cambridge Structural Database via "unusualness" $z$-scores.
- **Key Limitation Relative to Our Work:** Validates static 3D coordinates (e.g., X-ray refinements or docked poses) rather than coupling geometry $z$-scores with spectroscopic likelihoods to repair noisy 2D optical structure extractions.

### 28. Chemical Structure Curation ("Trust, But Verify" I & II)
- **Authors:** Denis Fourches, Eugene Muratov, and Alexander Tropsha
- **Year:** 2010 (and 2016)
- **Title:** *Trust, But Verify: On the Importance of Chemical Structure Curation in Cheminformatics and QSAR Modeling Research*
- **Venue / Identifier:** *Journal of Chemical Information and Modeling*, 50(7), 1189–1204 (`DOI: 10.1021/ci100176x`) [and *Trust, but Verify II: A Practical Guide to Chemogenomics Data Curation*, *JCIM* 2016, 56(7), 1243–1252, `DOI: 10.1021/acs.jcim.6b00129`]
- **Core Contribution:** Demonstrates that 0.1–10% structural errors—especially **silent stereochemical inversions, tautomer mismatches, and duplicate stereoisomer conflicts**—catastrophically degrade downstream QSAR/ML property prediction accuracy.
- **Key Limitation Relative to Our Work:** Standard curation pipelines only perform 2D valence/salt/tautomer normalization and duplicate flagging; they cannot *detect or fix* an inverted chiral wedge in a single literature-extracted record unless cross-checked against that compound's 3D physics and experimental spectra.

---

## Cluster 1D — Frontier LLM/VLM Chemistry & Multimodal Science Benchmarks (Papers 29–38)

### 29. ChemBench
- **Authors:** Adrian Mirza, Nawaf Alampara, Sreekanth Kunchapu, ..., Philippe Schwaller, and Kevin Maik Jablonka
- **Year:** 2025
- **Title:** *A framework for evaluating the chemical knowledge and reasoning abilities of large language models against the expertise of chemists*
- **Venue / Identifier:** *Nature Chemistry*, 17 (`DOI: 10.1038/s41557-025-01815-x`)
- **Core Contribution:** Evaluates frontier LLMs against human chemists across >2,700 curated chemistry questions spanning analytical, organic, physical, inorganic, and safety reasoning, revealing overconfident hallucinations on SMILES topology and isomer reasoning.
- **Key Limitation Relative to Our Work:** Predominantly text/SMILES-centric exam/expert QA; lacks real-document visual OCSR perception paired with 3D conformational strain and multi-nuclear NMR/HRMS diastereomer disambiguation from authentic organic synthesis SIs.

### 30. MolPuzzle
- **Authors:** Kehan Guo, Bozhao Nan, Yujun Zhou, Taicheng Guo, Zhichun Guo, Mihir Surve, Zhenwen Liang, Nitesh V. Chawla, Olaf Wiest, and Xiangliang Zhang
- **Year:** 2024
- **Title:** *Can LLMs Solve Molecule Puzzles? A Multimodal Benchmark for Molecular Structure Elucidation*
- **Venue / Identifier:** *Advances in Neural Information Processing Systems (NeurIPS 2024, Datasets and Benchmarks Track)* (`arXiv:2404.02360`)
- **Core Contribution:** Constructs ~234 multi-step structure elucidation puzzles (>18k–23k QA items) across 3 sequential stages—*Molecule Understanding* (degree of unsaturation/formula), *Spectrum Interpretation* (IR, MS, $^1\text{H}$/$^{13}\text{C}$ NMR images), and *Molecule Construction*—showing frontier VLMs (GPT-4o) fail severely on final structure assembly.
- **Key Limitation Relative to Our Work:** Uses clean, low-MW textbook-style achiral/simple molecules (`de novo` from spectra alone, without molecular structure images), ignoring real-world complex polycyclic stereoisomers, 3D steric strain physics, and cross-modal visual-diagram + SI-characterization reconciliation.

### 31. LAB-Bench
- **Authors:** Jon M. Laurent, Joseph D. Janizek, Michael Ruzo, Michaela M. Hinks, Michael J. Hammerling, Siddharth Narayanan, Manvitha Ponnapati, Andrew D. White, and Samuel G. Rodriques
- **Year:** 2024
- **Title:** *LAB-Bench: Measuring Capabilities of Language Models for Biology Research*
- **Venue / Identifier:** FutureHouse Technical Report (`arXiv:2407.10362`)
- **Core Contribution:** Introduces >2,400 multiple-choice questions with an explicit **abstention ("insufficient information") option** across practical research workflows (`LitQA2`, `FigQA`, `SuppQA`, `DbQA`, `ProtocolQA`, `SeqQA`), measuring both precision and coverage.
- **Key Limitation Relative to Our Work:** Biology-focused (no organic synthesis OCSR, 3D stereochemistry, or NMR/HRMS verification), though its **precision-coverage / calibrated abstention design** (`SuppQA` + `FigQA`) inspires rigorous selective-prediction metrics for our benchmark.

### 32. ChemVLM
- **Authors:** Junxian Li, Di Zhang, Xunzhi Wang, Zeying Hao, ..., Wanli Ouyang, Yuqiang Li, and Dongzhan Zhou
- **Year:** 2025
- **Title:** *ChemVLM: Exploring the Power of Multimodal Large Language Models in Chemistry Area*
- **Venue / Identifier:** *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2025)* (`arXiv:2408.07246`)
- **Core Contribution:** Trains a 26B ViT-MLP-InternLM2 chemistry VLM and introduces three evaluation suites: **MMChemOCR** (image-to-SMILES OCSR), **CMMU** (multimodal molecule understanding), and **MMCR-bench** (exam-style multimodal chemical reasoning).
- **Key Limitation Relative to Our Work:** Treats `MMChemOCR` as isolated 2D image-to-SMILES translation and `MMCR-bench` as high-school/college exam QA; evaluates zero 3D conformational energetics and never tests cross-modal reconciliation between noisy journal depictions and experimental NMR/HRMS data.

### 33. IsoBench
- **Authors:** Deqing Fu, Ruohao Guo, Ghazal Khalighinejad, Ollie Liu, Bhuwan Dhingra, Dani Yogatama, Robin Jia, and Willie Neiswanger
- **Year:** 2024
- **Title:** *IsoBench: Benchmarking Multimodal Foundation Models on Isomorphic Representations*
- **Venue / Identifier:** *First Conference on Language Modeling (COLM 2024)* (`arXiv:2404.01266`)
- **Core Contribution:** Demonstrates that frontier VLMs systematically underperform by 10–30+ points on visual/image representations compared to isomorphic text/symbolic representations across chemistry, math, games, and graph algorithms.
- **Key Limitation Relative to Our Work:** Chemistry subset uses clean RDKit-rendered 2D depictions of basic property/isomer questions; does not examine **complementary (non-isomorphic) real-world modalities** where visual depictions are degraded while spectroscopic text provides orthogonal 3D stereochemical constraints.

### 34. MMMU (and MMSci)
- **Authors:** Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, ..., Huan Sun, Yu Su, and Wenhu Chen
- **Year:** 2024
- **Title:** *MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI*
- **Venue / Identifier:** *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024)*, pp. 9556–9567 (`arXiv:2311.16502`) [extended to PhD figure-caption comprehension in Zekun Li et al., *MMSci: A Dataset for Graduate-Level Multi-Discipline Multimodal Scientific Understanding*, `arXiv:2407.04903`, 2024]
- **Core Contribution:** Benchmarks college- and graduate-level multimodal reasoning across 30 subjects including chemistry diagrams, plots, and reaction schemes.
- **Key Limitation Relative to Our Work:** Multiple-choice QA susceptible to superficial elimination shortcuts and pretraining memorization of textbook problems; lacks executable cheminformatics/quantum-mechanical ground truth or stereoisomer-level structural output verification.

### 35. SUPERChem (with Humanity's Last Exam & FrontierScience)
- **Authors:** Zehua Zhao, Zhixian Huang, Junren Li, Siyu Lin, ..., Zhen Gao, and Jian Pei
- **Year:** 2025
- **Title:** *SUPERChem: A Multimodal Reasoning Benchmark in Chemistry*
- **Venue / Identifier:** `arXiv:2512.01274` [alongside Long Phan, ..., Dan Hendrycks, *Humanity's Last Exam*, `arXiv:2501.14249`, 2025; and Miles Wang et al., *FrontierScience: Evaluating AI's Ability to Perform Expert-Level Scientific Tasks*, `arXiv:2601.21165`, 2026]
- **Core Contribution:** Curates 500 Olympiad/graduate-level multimodal & text chemistry problems with step-by-step **Reasoning Path Fidelity (RPF)** checkpoints, showing frontier reasoning VLMs still bottleneck on fine-grained structural perception and multi-step stereochemical deduction.
- **Key Limitation Relative to Our Work:** Hand-crafted competition/PhD problems (~500 total) rather than scalable, physics-and-spectroscopy-grounded extraction/reasoning tasks anchored to authentic post-cutoff experimental characterization records and 3D strain energetics.

### 36. Mol-Instructions (and ChemLLM)
- **Authors:** Yin Fang, Xiaozhuan Liang, Ningyu Zhang, Kangwei Liu, Rui Huang, Zhuo Chen, Xiaohui Fan, and Huajun Chen
- **Year:** 2024
- **Title:** *Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset for Large Language Models*
- **Venue / Identifier:** *The Twelfth International Conference on Learning Representations (ICLR 2024)* (`arXiv:2306.08018`) [alongside Di Zhang et al., *ChemLLM: A Chemical Large Language Model*, `arXiv:2402.06852`, 2024]
- **Core Contribution:** Large-scale instruction tuning dataset (>2M prompts) covering molecule-oriented, protein-oriented, and biomolecular text tasks for LLMs.
- **Key Limitation Relative to Our Work:** Purely 1D SMILES/SELFIES + text without molecular images, 3D geometries, or spectroscopic modalities; propagates any upstream database stereochemical curation errors directly into LLM supervision.

### 37. LlaSMol (SMolInstruct)
- **Authors:** Botao Yu, Frazier N. Baker, Ziqi Chen, Xia Ning, and Huan Sun
- **Year:** 2024
- **Title:** *LlaSMol: Advancing Large Language Models for Chemistry with a Large-Scale, Comprehensive, High-Quality Instruction Tuning Dataset*
- **Venue / Identifier:** *First Conference on Language Modeling (COLM 2024)* (`arXiv:2402.09391`)
- **Core Contribution:** Curates **SMolInstruct** (>3M samples across 14 chemistry tasks including name conversion, property prediction, forward synthesis, and retrosynthesis) with canonicalized SMILES evaluation.
- **Key Limitation Relative to Our Work:** Text/SMILES-only modality; lacks visual structure recognition, 3D conformational thermodynamics, and analytical characterization (NMR/MS) grounding.

### 38. ChemDFM-X (and MolCA)
- **Authors:** Zihan Zhao, Bo Chen, Jingpiao Li, Lu Chen, ..., Xin Chen, and Kai Yu
- **Year:** 2024
- **Title:** *ChemDFM-X: Towards Large Multimodal Model for Chemistry*
- **Venue / Identifier:** `arXiv:2409.13194` [extending Zihan Zhao et al., *Developing ChemDFM as a large language foundation model for chemistry*, `arXiv:2401.14818`, 2024; and Zhiyuan Liu et al., *MolCA: Molecular Graph-Language Modeling with Cross-Modal Projector and Uni-Modal Adapter*, *EMNLP 2023*, `arXiv:2310.12798`]
- **Core Contribution:** Aligns 5 modalities—2D graphs, 3D conformations, 2D molecular images, MS/MS spectra, and IR spectra—into a single multimodal LLM via modality-specific encoders and cross-modal projectors across 7.6M instruction pairs.
- **Key Limitation Relative to Our Work:** **Omits NMR spectroscopy entirely** (the single indispensable modality for stereochemistry and diastereomer assignment!), treats each modality via separate pairwise instruction prompts rather than joint Bayesian physical-spectroscopic hypothesis verification, and does not enforce 3D strain or exact-mass constraints at inference.

---

## Cluster 1E — Supporting Information (SI) Automated Extraction & Multimodal Record Verification (Papers 39–42)

### 39. ChemDataExtractor & ChemDataExtractor 2.0
- **Authors:** Juraj Mavračić, Callum J. Court, Taketomo Isazawa, Stephen R. Elliott, and Jacqueline M. Cole
- **Year:** 2021 (and 2016)
- **Title:** *ChemDataExtractor 2.0: Autopopulated Ontologies for Materials Science*
- **Venue / Identifier:** *Journal of Chemical Information and Modeling*, 61(9), 4280–4289 (`DOI: 10.1021/acs.jcim.1c00446`) [building on Matthew C. Swain and Jacqueline M. Cole, *ChemDataExtractor: A Toolkit for Automated Extraction of Chemical Information from the Scientific Literature*, *JCIM* 2016, 56(10), 1894–1904, `DOI: 10.1021/acs.jcim.6b00207`]
- **Core Contribution:** Document-level NLP pipeline linking chemical entity mentions, compound labels, tables, and experimental property/spectral blocks (including regex/grammar parsers for $^1\text{H}/^{13}\text{C}$ NMR and melting points).
- **Key Limitation Relative to Our Work:** Extracts text-reported NMR peak lists and links compound labels, yet treats extracted spectra as passive metadata strings—never running forward 3D NMR/strain models to verify whether the linked chemical structure or stereochemistry is physically consistent with the extracted spectrum.

### 40. The Open Reaction Database (ORD)
- **Authors:** Steven M. Kearnes, Michael R. Maser, Michael Wleklinski, Amber Kast, Abby G. Doyle, Spencer D. Dreher, John M. Hawkins, Klavs F. Jensen, and Connor W. Coley
- **Year:** 2021
- **Title:** *The Open Reaction Database*
- **Venue / Identifier:** *Journal of the American Chemical Society*, 143(45), 18820–18826 (`DOI: 10.1021/jacs.1c09820`)
- **Core Contribution:** Establishes a structured protobuf schema and open repository (>2M reactions) supporting reaction conditions, product identifiers, and analytical characterization fields (NMR, MS, yield, ee/dr).
- **Key Limitation Relative to Our Work:** Literature-mined entries in ORD (mostly USPTO text-mined) overwhelmingly lack verified 3D stereochemical product validation against SI analytical data, creating a direct downstream ecosystem need for physically and spectroscopically verified extraction.

### 41. NMRexp (and NMRExtractor / NMRBank)
- **Authors:** Jun-Jie Wang, Yongqi Jin, Chen-Yu Zhi, Yu-Jie Liu, Xu-Hao Huang, Fanjie Xu, Xiaohong Ji, Xi Fang, Haoyi Tao, Weinan E, Linfeng Zhang, Guolin Ke, and Rong Zhu
- **Year:** 2025
- **Title:** *NMRexp: A database of 3.3 million experimental NMR spectra*
- **Venue / Identifier:** *Scientific Data*, 12 (`DOI: 10.1038/s41597-025-06245-5`) [alongside Qinggong Wang et al., *NMRExtractor: leveraging large language models to construct an experimental NMR database from open-source scientific publications*, *Chemical Science* 2025, `DOI: 10.1039/d4sc08802f`, releasing `NMRBank` (~225k records)]
- **Core Contribution:** Parses ~200,000 open-access organic synthesis Supporting Information PDFs (2010–2024) by combining OCSR (MolScribe) on SI structure depictions with LLM/rule-based NMR text extraction to build a 3.3M-record experimental NMR dataset across 6 nuclei ($^1\text{H}, {}^{13}\text{C}, {}^{19}\text{F}, {}^{31}\text{P}, {}^{29}\text{Si}, {}^{11}\text{B}$).
- **Key Limitation Relative to Our Work:** **Empirical smoking gun for our entire project!** In NMRexp's own manual technical validation, OCSR achieves **98% accuracy on 2D molecular skeletons** but **drops to 83% accuracy on stereochemistry even for single-stereocenter molecules** (a **17% silent stereochemical error rate**, and worse on multi-stereocenter diastereomers!). Because NMRexp and NMRExtractor do *not* use the extracted NMR/HRMS data + 3D Boltzmann energetics to rerank/repair OCSR stereocenters, hundreds of thousands of stereochemically inverted records silently contaminate modern AI4Chemistry training sets!

### 42. TRACE-Chem (ChemEvidence) — *Closest Competing Work*
- **Authors:** Jiaqi Liu, Changshuai Wang, and Hanshen Li
- **Year:** 2026
- **Title:** *TRACE-Chem: Symmetry-Aware Counterfactual Reasoning over Typed Evidence Graphs for Verifiable Multimodal Chemical Record Extraction*
- **Venue / Identifier:** *Symmetry*, 18(9), 1437 (`DOI: 10.3390/sym18091437`)
- **Core Contribution:** Introduces a Symmetric–Asymmetric Evidence Graph (SAEG) and Counterfactual Localization (CLDR) with a `ChemEvidence` verifier checking cross-field consistency across text names (OPSIN), top-1 OCSR (MolScribe/MolNexTR), formula/adduct $m/z$, 2D graph-based NMR shift predictors, and CFM-ID 4.0 MS/MS across 124 open-access organic synthesis papers (1,852 compounds), improving canonical record F1 by +9.7 pp (to 84.3%) and lowering ECE from 0.243 to 0.052.
- **Key Limitation Relative to Our Work:** Four fundamental blind spots separate TRACE-Chem from `001-phys-spec-ocsr-bench`:
  1. **Zero 3D Physics or Thermodynamic Strain:** Uses no 3D conformer generation, no MLIPs (MACE-OFF23 / AIMNet2 / xTB), and no steric strain / ring-junction / anti-Bredt geometry checks.
  2. **Diastereomer-Blind 2D Spectral Verification:** Relies on 2D topological GNN NMR shift predictors and 2D MS fragmenters (CFM-ID 4.0), which share identical 2D connectivity, molecular formula, and exact HRMS $m/z$ across diastereomers and miss 3D through-space anisotropic/$\gamma$-gauche shieldings and Karplus $^3J_{\text{HH}}$ dihedral couplings.
  3. **Field-Masking vs. Stereoisomer/Beam Hypothesis Space Search:** Masks conflicting extracted fields (relying on OPSIN IUPAC names when OCSR fails), which fails whenever a scheme structure (**3a**) lacks a full `(1R,2S,4R)` IUPAC name in prose; it never enumerates uncertain-wedge diastereomer ensembles or OCSR beam candidates for 3D Boltzmann + DP4-style probabilistic reranking.
  4. **No Frontier Multimodal AI4Science Benchmark:** Evaluates an internal extraction wrapper on 124 papers rather than constructing a leakage-controlled, multi-task diagnostic benchmark testing VLMs, reasoning models, and OCSR pipelines on coupled visual-thermodynamic-spectroscopic reasoning.

---

## Gap & Creativity Synthesis Matrix

| Method / Benchmark Family | Representative Works | 2D Visual OCSR | 3D Conformer & MLIP Strain ($\Delta E_{\text{strain}}$) | Exact HRMS / Formula Hard Filter | 3D Stereosensitive NMR ($\delta_C, \delta_H, {}^3J_{\text{HH}}$) | Epimer / Beam Reranking & Repair | Frontier VLM / LLM Diagnostic Benchmark |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure 2D Vision OCSR** | DECIMER.ai (#1), MolScribe (#2), MolGrapher (#3), MolNexTR (#7), AtomLenz (#8), MolSight (#9) | **Yes** | No | No | No | No (top-1 2D pixels only) | Narrow 2D image$\to$SMILES only |
| **3D QM/ML NMR Diastereomer Assignment** | DP4 (#11), DP4+ (#12), ML-J-DP4 (#13), DP4-AI (#14), CASCADE (#15), IMPRESSION (#17) | No | Partial (slow DFT or MMFF) | Assumed known | **Yes** (DFT or 3D GNN) | Over user-supplied isomers only | No |
| **Spectra-Only *De Novo* Elucidation** | Alberts et al. (#18), MassSpecGym (#19), DiffMS (#20), DeepSAT (#21), MolPuzzle (#30) | No | No | Partial (DiffMS) | No (mostly 2D achiral SMILES) | No | Spectra$\to$2D SMILES only |
| **3D Geometry & Strain Sanity Suites** | PoseBusters (#22), GenBench3D (#23), Auto3D (#24), McDermott/Garg (#26), Mogul (#27) | No | **Yes** | No | No | No | Docking / 3D generative only |
| **Frontier Chemistry LLM/VLM Benchmarks** | ChemBench (#29), LAB-Bench (#31), ChemVLM (#32), IsoBench (#33), SUPERChem (#35), ChemDFM-X (#38) | Partial (clean 2D) | No | No | No (ChemDFM-X omits NMR!) | No | Exam QA / isolated tasks |
| **SI Extraction & 2D Record Checking** | OpenChemIE (#6), NMRexp (#41), TRACE-Chem (#42) | **Yes** (off-the-shelf) | **No** | Yes (TRACE-Chem) | **No** (2D graph NMR only) | Field masking only (no 3D epimer search) | No |
| **`001-phys-spec-ocsr-bench` (Ours)** | **Thermodynamically & Spectroscopically Informed OCSR + Benchmark** | **Yes** (degraded/wild SI + $SO(2)$-aware) | **Yes** (ETKDGv3 + MACE-OFF / xTB continuous $\Delta E_{\text{strain}}$) | **Yes** (exact monoisotopic mass / adduct / $^{13}\text{C}$ symmetry count) | **Yes** (Boltzmann-weighted 3D GNN shifts + Karplus $^3J_{\text{HH}}$ + unassigned DP4-AI matching) | **Yes** (joint Bayesian posterior over beam + uncertain-wedge epimers) | **Yes** (calibrated abstention + counterfactual ablation tracks) |

---

## Strategic Takeaways & Concrete Creative Upgrades for `001-phys-spec-ocsr-bench`

1. **Empirical Motivation Anchor (`NMRexp`, *Sci. Data* 2025):** Cite `NMRexp` (#41) directly in Paragraph 1: state-of-the-art OCSR (MolScribe) on 200,000 real Supporting Information PDFs achieves **98% 2D skeleton accuracy** yet drops to **83% stereochemical accuracy even on single-stereocenter molecules** (17% silent chiral/diastereomeric corruption across 3.3M extracted NMR records!). Worse, `Fourches et al.` (#28) prove that silent stereochemical inversions poison downstream ML models, while `PoseBusters` (#22) and `GenBench3D` (#23) show 2D RDKit sanitization passes severely strained 3D geometries.
2. **Crisp Differentiation Against `TRACE-Chem` (*Symmetry* 2026) & `MolSight` (*AAAI* 2026):**
   - `MolSight` (#9) proves OCSR stereochemistry is an acute frontier bottleneck (`Stereo-200K`), but hits an **information-theoretic pixel ceiling** on low-DPI/compressed literature scans where wedge/dash contrast or bridged-ring perspective is visually lost.
   - `TRACE-Chem` (#42) checks 2D cross-field consistency (OPSIN name vs. OCSR vs. formula vs. 2D graph NMR), which **completely fails on diastereomers without full IUPAC stereochemical prose names** because diastereomers have *identical* formulas, *identical* HRMS $m/z$, and *identical* 2D molecular graphs! Resolving diastereomeric wedge inversions *requires* **3D conformer ensembles + 3D thermodynamic strain + 3D through-space NMR ($\delta_C, \delta_H$) + Karplus dihedral $^3J_{\text{HH}}$ couplings**.
3. **Continuous Strain Physics Over Brittle Rules (`McDermott, Houk & Garg`, *Science* 2024):**
   - Replace binary "valid/invalid" topological rules (like rigid Bredt's rule SMARTS) with a **continuous Boltzmann / relative strain energy prior** ($\Delta E_{\text{strain}}$ via ETKDGv3 $\to$ MACE-OFF23 / GFN2-xTB) across the stereoisomer family, calibrated to the experimental context (compounds characterized by room-temperature solution NMR in SI must be kinetically/thermodynamically isolable in solution, sharply penalizing $+15\text{ to }+40\text{ kcal/mol}$ inverted ring-junction hallucinations!).
4. **Unassigned Peak-Set Matching + Symmetry Counting (`DP4-AI` + $^{13}\text{C}$ Signal Count):**
   - Because SI NMR peak lists are *unassigned* ($\delta_C$ list, $\delta_H$ multiplet/$J$ list), pair **Hungarian/optimal-transport bipartite peak matching** (`DP4-AI` #14) over Boltzmann-weighted 3D GNN shift predictions (`CASCADE` #15) with **3D point-group / topological $^{13}\text{C}$ signal-count symmetry checks** (e.g., *meso*/*$C_2$-symmetric* vs. *$C_1$-asymmetric* diastereomers immediately differ in total $^{13}\text{C}$ peak count—a zero-cost, exact discriminator between *cis* and *trans* cyclic stereoisomers!).
5. **Enantiomer vs. Diastereomer Epistemic Honesty ($SO(2)$ vs. $O(2)$ + Achiral NMR Gauge Invariance):**
   - Explicitly formalize the fundamental physical symmetry law in our method and benchmark: **achiral solution NMR + scalar 3D energy are $SE(3)$-invariant ($E(3)$-invariant for enantiomers)**—they distinguish *diastereomers*, *epimers*, *regioisomers*, and *cis/trans* ring fusions, whereas *absolute configuration* ($R,R$ vs. $S,S$ enantiomers) is observable ONLY visually ($SO(2)$-equivariant wedge perception, Weiler & Cesa #10) or via chiroptical/crystallographic text (`[\alpha]_D`, X-ray Flack parameter). Baking this exact symmetry decomposition into both our reranker and a dedicated **Enantiomer-vs.-Diastereomer Epistemic Diagnostic Track** (with `LAB-Bench`-style calibrated abstention when pixels are destroyed and only achiral NMR is present!) will stand out immediately to both ML reviewers and physical organic chemists.
