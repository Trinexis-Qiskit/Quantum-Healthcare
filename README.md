# Q-MetaboX: Reference-Free Metabolite and Unknown-Compound Identification
### IBM Fall Fest 2026 | Trinexis x IEEE | Domain: Healthcare / Life Sciences (Pharma and Metabolomics) | Participant Guide

---

## 1. Problem statement

### 1.1 Context
Mass spectrometry is how pharmaceutical, clinical and natural-product laboratories see the small molecules in a sample. In **LC-MS/MS** (liquid chromatography tandem mass spectrometry) each detected molecule is broken into fragments, producing a **fragmentation spectrum**: a list of fragment masses (m/z) with intensities. Identifying *which molecule* produced a spectrum is the bottleneck of three real workflows:

1. **Unknown impurities and degradants** in drug development (what is this peak that was not in the specification?).
2. **Unknown metabolites** in bioanalysis and metabolomics (which biotransformation product is this?).
3. **Natural-product and biomarker discovery** (most molecules in a plant, microbial or biofluid extract have no reference spectrum).

The classical answer is to match the measured spectrum against a library of reference spectra. That fails exactly when it matters: for a molecule that is not in the library. **Reference-free identification** instead predicts structural evidence from the spectrum and ranks candidate *structures* from a structure-only database, without ever looking up a reference spectrum of the query molecule. An analyst then reviews a short ranked list (here: 25 structures) instead of starting from nothing.

This is **candidate ranking**, not de novo structure generation, and it is a research prototype, not a validated clinical or regulatory tool.

### 1.2 Your task
Build and compare three rankers (one classical, two hybrid quantum-classical) that answer:

> Given **all** MS/MS spectra of one unknown molecule, can a ranker place the correct 2D structure near the top of a list of at most 25 unique SMILES drawn from a structure-only candidate catalog, on molecules it has never seen, and do real-hardware quantum components add anything to a strong classical ranker?

| ID | Method | Type |
|---|---|---|
| **M1** | Spectrum-to-fingerprint models plus a LambdaRank candidate ranker | Classical ML |
| **M2** | Quantum-kernel SVM (QSVC) re-ranker, kernel entries measured on a real IBM quantum computer | Hybrid, kernel |
| **M3** | QAOA feature selection, then a hardware-trained Variational Quantum Classifier (VQC) re-ranker | Hybrid, variational |

You run the quantum parts on a **real IBM quantum computer** through IBM Cloud within about **10 minutes of QPU time**, on a small cohort of molecules.

### 1.3 Ground rules
- **No quantum simulation, ever.** Forbidden: `qiskit_aer`, `Aer`, `AerSimulator`, `Statevector*`, `DensityMatrix`, `BasicSimulator`, `Fake*Backend`, `StatevectorSampler`, `StatevectorEstimator`, `NumPyMinimumEigensolver` as a stand-in for QAOA, tensor-network or noisy simulators, mocked or hand-typed counts, cached demo outputs.
- Every quantum number (kernel entry, QAOA objective, VQC loss, VQC prediction) must come from measurements returned by a real IBM backend. A classical optimiser may *propose* circuit parameters; it may never *invent* a circuit's result. Exact classical optima and closed-form formulas are allowed only as clearly labelled **reference controls**, never as quantum data.
- Local CPU is allowed for pandas, scikit-learn, LightGBM, RDKit, feature engineering, QUBO construction, transpilation and plotting.
- **Use only the files in `data/`.** No external spectra, structures, fingerprints, embeddings, pretrained models or other datasets (see `data/README.md`).
- If something is unavailable (IBM token, quota, backend), stop that phase, mark it `BLOCKED_<REASON>` (for example `BLOCKED_IBM_ACCESS`, `BUDGET_EXHAUSTED`) and say so. Never fill the gap with fake or simulated data.
- Never put API keys in files or notebooks. Use environment variables.
- Never use a label, an identifier or a label-derived column as a model input (Section 3.4).

### 1.4 The problem in practice
Picture the analytical lab of a pharma company. An unknown peak was fragmented under several conditions, giving, say, six spectra of the same molecule (different adducts such as `[M+H]+` and `[M+Na]+`, different collision energies, maybe different instruments). Your system must turn those spectra into **one ranked list**.

The pipeline has four stages, and each one can fail independently:

| Stage | What happens | How it can fail | What you report |
|---|---|---|---|
| 1. Catalog | The correct structure must exist in `data/structure_catalog.parquet` | Not in the catalog | Catalog coverage |
| 2. Candidate pool | Convert precursor m/z and adduct to a neutral mass; keep catalog structures within a mass tolerance | Mass tolerance too tight, adduct inconsistent across spectra | **Pool recall** (truth in pool) |
| 3. Ranking | Score every pool candidate against the spectra and sort | Right structure in the pool but ranked low | MRR@25, Top-1, Hit@5, Hit@25 |
| 4. Quantum re-rank | Reorder only the top-K shortlist with a hardware-measured score | Truth not in the shortlist; noisy hardware | Shortlist coverage, paired change versus M1 |

Why this is hard: many catalog structures share almost the same mass, so mass alone leaves a long list; fragment patterns are noisy and depend on adduct, instrument and collision energy; and you must score **molecules you have never seen**, so a model that memorises training molecules is useless.

Why the quantum part is small: you have about 10 minutes of real QPU time. Quantum models therefore only **re-order the top-2 shortlist** of the classical ranker on a small, diversity-balanced cohort of held-out internal-test molecules, using 4 quantum features and 2 to 4 qubits. This is a feasibility experiment on real hardware, not a production system. The honest question is how the quantum methods behave next to a strong classical baseline on the same molecules, and what hardware noise does to them.

### 1.5 Solution roadmap
Work through these steps in order. Each produces something the next one needs.

| Step | What you do | What you produce | Details |
|---|---|---|---|
| 1 | Set up the notebook: one `config` cell, credentials from environment variables, simulator-ban scan | Config cell, pinned versions | 1.7, 4.4 |
| 2 | Verify the data checksums, audit the schema, build the leakage register | Hashes, schema audit, feature register | 3.1 to 3.4 |
| 3 | Explore the data (spectra per molecule, adducts, instruments, masses, peaks) | EDA figures | 3.5 |
| 4 | Adduct-aware neutral mass, candidate pools, pool recall on training molecules | Mass tolerance, pool statistics | 3.6, 4.1 |
| 5 | Train M1 on the 400 training molecules; choose settings by group cross-validation inside the training set | Frozen M1, controls, ablations | 4.1 |
| 6 | Evaluate on the internal 100, then **freeze** (hashes, config) and write top-25 predictions for the unlabelled final 150 | Internal-100 metrics with bootstrap intervals; final-150 predictions file | 3.5, 6 |
| 7 | Build the frozen quantum cohort: queue, top-2 shortlists, train anchors, scaler | Frozen quantum inputs | 3.8 |
| 8 | QPU preflight: quota, backend, physical qubits, confirm it is not a simulator; tiny calibration job | Backend and layout record | 5.3 |
| 9 | QAOA on 4 qubits to choose 2 of 4 quantum features | Measured counts, selected mask | 4.3 |
| 10 | M2: measure Gram and cross-kernel on hardware for a **basic** and an **improved** feature map, fit the SVM | Kernel matrices traced to job IDs | 4.2 |
| 11 | M3: hardware-in-the-loop VQC training and inference, **BEFORE** and **AFTER** circuits | Loss versus QPU calls, paired predictions | 4.3, 5 |
| 12 | Optional improvement round (pairwise keep/swap, error mitigation, hardware-native map) | Second set of measured kernels | 4.5 |
| 13 | Re-query every job ID, evaluate all methods on identical molecules with paired bootstrap, write findings | Benchmark table, charts, findings | 6, 7 |
| 14 | Single-molecule demo: all spectra of one molecule in, top-25 SMILES out for M1, M2, M3 | Annotated spectrum and ranked structures | 2.2 |

Order the QPU work so cheap, informative jobs come first (tiny calibration, then QAOA, then kernels, then VQC) and keep at least 20 % of the budget in reserve for retries. Save each job ID the moment it is submitted.

How to decide when you finish:
- If a quantum variant does not improve on M1, report it as measured. A well-explained negative result is acceptable; an unsupported "improved" claim is not.
- With a few dozen quantum molecules, a difference of one or two molecules is not evidence. Report intervals and paired deltas; do not rank methods on noise.
- Never describe quantum advantage unless your paired intervals support it.

### 1.6 Common mistakes to avoid
- Splitting by spectrum instead of by molecule (all spectra of a molecule must stay on one side).
- Using `molecule_id`, `spectrum_id`, `precursor_error_ppm`, `molecular_formula`, `ingest_lib`, `adduct_orig`, `canonical_identity` or any structure-derived value as a query-time model input.
- Changing a model after you have written the final-150 predictions, or retuning on the internal 100 after you have looked at its scores without labelling the change post-hoc. The final-test answers are held by the organisers; never try to recover them.
- Fitting scalers, clipping bounds, percentile ranks, thresholds or the ranker on final-test molecules.
- Pruning the candidate catalog with split knowledge (for example removing training structures).
- Reporting a one-spectrum-per-row score. The official unit is the **molecule**.
- Inserting the true structure into a candidate list or shortlist. Quantum scores may only **re-order** existing shortlist candidates (oracle insertions must equal zero).
- Comparing 150-molecule classical numbers with 40-molecule quantum numbers as if they were equivalent.
- Using in-sample M1 scores as a feature for quantum training pairs without cross-fitting.
- Reading SVM decision scores as probabilities.
- Claiming a kernel diagonal was measured when you set it to 1 by definition.
- Resubmitting a job after a notebook restart instead of retrieving it with `service.job(job_id)`, which wastes QPU time.
- Using Qiskit Machine Learning's default `VQC` or `SamplerQNN` without proving its sampler is real hardware (the defaults are local simulators).
- Filling a gap (no token, no quota) with fake or cached outputs instead of marking the phase `BLOCKED_<REASON>`.

### 1.7 Get your IBM Quantum account (10 minutes of QPU time)
The free **Open Plan** on IBM Quantum Platform gives you about 10 minutes of real QPU time per rolling 28-day window, which is the budget this challenge is built around. Do this before the event because account creation and identity checks can take a while. IBM changes its portal from time to time, so if a screen differs from the steps below, follow IBM's current on-screen instructions.

**Create the account**
1. Open https://quantum.cloud.ibm.com and choose **Sign up** (or **Log in** with an existing IBMid).
2. Register with an email address, verify it and accept the terms. This creates an IBM Cloud account.
3. Choose the **Open Plan** and create an instance. It needs no payment details; do not pick a paid plan.

**Collect the three things your notebook needs**

| Item | Where to find it | Notes |
|---|---|---|
| API key | IBM Quantum Platform dashboard (API key panel) or https://cloud.ibm.com/iam/apikeys | Shown once when created; copy it right away |
| Instance CRN | Open the instance on the Instances page and copy its CRN | Long identifier starting with `crn:v1:` |
| Channel | `ibm_quantum_platform` | Name used by recent `qiskit-ibm-runtime`; check against your installed version |

**Connect without putting secrets in files**
```bash
export QISKIT_IBM_TOKEN="<your API key>"
export QISKIT_IBM_INSTANCE="<your instance CRN>"
export QISKIT_IBM_CHANNEL="ibm_quantum_platform"
```
```python
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()          # reads the environment variables
print(service.usage())                    # remaining QPU time; record this number
```
If `service.usage()` fails, that is `BLOCKED_IBM_ACCESS`: stop the quantum phases and report it.

**Using the 10 minutes well**
- QPU time is billed for execution only; queue waiting does not count. Develop and unit-test all classical code first, because the allowance resets on a rolling window and test runs consume it.
- The key is a password: never commit it, never paste it into a cell, and delete it in IBM Cloud if it is ever exposed.

**Install the environment (Python 3.12 recommended)**
```text
qiskit==2.5.0
qiskit-ibm-runtime==0.50.0
rdkit==2026.9.1
pyarrow==24.0.0
pandas==3.0.3
numpy==2.5.1
scikit-learn==1.9.0
lightgbm==4.7.0
matplotlib==3.11.0
pylatexenc==2.11
joblib==1.5.3
ipykernel==7.3.0
nbformat==5.10.4
nbconvert==7.17.1
```
Newer or older versions are acceptable if you print them in the notebook and say so. RDKit versions can differ slightly in tautomer canonicalisation; record yours.

---

## 2. Expected outcome

### 2.1 What you must submit
1. **One executed Jupyter notebook** (`.ipynb`, outputs saved, runs top to bottom without errors) containing the full classical and quantum workflow.
2. **Evidence files**: IBM job IDs, raw measured counts, kernel matrices, metric tables, per-molecule ranked lists, figures, the saved circuits (QPY) and a usage ledger.
3. **A findings section** at the end of the notebook stating which method you would choose and why, backed by your own numbers, including negative results.
4. Predictions as CSV: `predictions_final_test_150_M1.csv` (required: one row per final-test `molecule_id`, column `smiles` with up to 25 canonical, unique, semicolon-separated SMILES, best first, written from your frozen M1) plus one CSV per method for the internal 100 or the quantum cohort. The organisers score the final-test file; no answers are provided to participants.
5. The repository deliverables listed in `CONTRIBUTING.md` (README, `src/`, report, slides, demo video link).

### 2.2 What your solution must demonstrate

| # | Outcome | How it is judged |
|---|---|---|
| 1 | A leakage-safe data pipeline: checksums, schema audit, feature register, molecule-level split, frozen configuration | Printed checks, hashes, documented exclusions |
| 2 | One classical ML model (M1) with controls and ablations scored on the internal 100 (labelled) and used to predict the unlabelled final 150 | MRR@25, Top-1, Hit@5, Hit@25, pool recall, bootstrap intervals |
| 3 | Two hybrid quantum models (M2 kernel, M3 QAOA plus VQC) on a real IBM backend | Job IDs, raw counts, QPU seconds, real backend name |
| 4 | A fair comparison on **identical molecules and candidates** with identical decision rules | Same molecule IDs, same shortlists, paired bootstrap deltas versus M1 |
| 5 | A circuit **BEFORE/AFTER** comparison on the same backend, molecules, shots and task, with all circuits **created in your notebook and printed** (logical and transpiled) | Printed diagrams, depth and two-qubit counts, paired predictions |
| 6 | Honest interpretation: every insight matches printed numbers | Insight text versus outputs; no unsupported "improved" claims |
| 7 | Molecule-level top-25 output and a **single-molecule demo** (all spectra of one molecule to ranked structures for M1, M2 and M3) | Annotated spectrum, ranked SMILES, drawn structures, status labels |
| 8 | An end-goal scorecard comparing all methods on cost, accuracy, latency and risk | Final comparison table and chart |

### 2.3 Target to aim for
- **M1** should clearly beat the three classical controls (random, mass-only, fingerprint-only) on the internal 100; say by how much and with which interval.
- **Pool recall** is an upper bound on every method: report it and explain the molecules lost at the pool stage.
- Quantum methods are judged on **correctness of execution and honesty of analysis**, not on beating M1. A well-measured, well-explained result that matches or falls short of the classical ranker is a valid outcome.
- Do not claim quantum advantage; the quantum cohort is far too small to support it.

---

## 3. Data

All data are in the `data/` folder of this repository. **Read `data/README.md` first**: it holds the full data card (source, licence, schema, statistics, caveats).

### 3.1 Source (the only allowed source)
Experimental LC-MS/MS spectra from a public small-molecule tandem mass spectrometry collection (CC BY-NC 4.0, non-commercial use only). The files in `data/` are a prepared 650-molecule subset plus a structure-only candidate catalog; nothing needs to be downloaded from anywhere else. Log the SHA-256 of every file (`data/SHA256SUMS.txt`).

### 3.2 Files

| File | Rows | Content | Used for |
|---|---|---|---|
| `train_400.parquet` | 3,423 spectra, 400 molecules | Spectra plus labels | Fit everything |
| `internal_test_100.parquet` | 1,127 spectra, 100 molecules | Spectra plus labels | Model selection and development |
| `final_test_150_inputs.parquet` | 1,620 spectra, 150 molecules | **Spectra only, no labels** | Final test: predict and submit; scored by the organisers |
| `structure_catalog.parquet` | 274,195 structures | SMILES, exact mass, formula, descriptors | Candidate bank |
| `identity_manifest.parquet` | 650 molecules | Split, polarity, instrument group, stratum | Quantum queue, EDA |

### 3.3 Task definition and official scoring
- **Unit of prediction:** one molecule (all spectra sharing a `molecule_id`).
- **Output:** a ranked list of at most 25 valid, unique SMILES from the catalog.
- **Correct** means the tautomer-canonical InChIKey14 of the predicted SMILES equals the molecule's `canonical_identity`:
```python
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
_taut = rdMolStandardize.TautomerEnumerator()
def identity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToInchiKey(_taut.Canonicalize(mol))[:14]
```
- **Primary metric:** MRR@25 = mean over molecules of `1 / rank` of the first correct structure (0 if it is not in the top 25). Secondary: Top-1, Hit@5, Hit@25.

### 3.4 Features: what you may and may not use

| Group | Columns | Rule |
|---|---|---|
| **Query inputs** | `ms2_mzs`, `ms2_normalized_intensities`, `base_peak_intensity`, `adduct`, `ionization_mode`, `instrument_type`, `precursor_mz`, `collision_energy_ev`, `collision_energy_orig`, `collision_energy_orig_units` | The only allowed model inputs on every split |
| **Identifiers** | `molecule_id`, `spectrum_id` | Grouping only; random and meaningless |
| **Candidate side** | `structure_catalog.parquet` columns | Allowed for scoring a candidate; never contains query information |
| **Labels and audit** | `normalized_smiles`, `inchikey`, `inchikey14`, `canonical_identity`, `molecular_formula`, `ingest_lib`, `adduct_orig`, `precursor_error_ppm`, `num_peaks`, `split` | Training targets, grouping and reporting only. **Never** a query-time input |

Publish your feature register (every column, its group, your decision and why) in the notebook.

### 3.5 Cohorts and how they are used

| Cohort | Molecules | Use |
|---|---|---|
| Train | 400 | Fit scalers, fingerprint models, ranker, quantum anchors and QUBO statistics. Use group k-fold over **molecules** inside it. |
| Internal test | 100 (labelled) | Your evaluation set for M1, the controls and the quantum cohort. Choose hyperparameters by cross-validation inside the training set so this cohort stays clean. |
| Final test | 150 (unlabelled) | Freeze your models, write top-25 predictions, submit them. Never used for fitting, scaling or tuning. |
| Quantum cohort | about 40 of the internal 100 | Small matched cohort for M1/M2/M3 comparison on hardware (Section 3.8) |

The split is **by molecule**. Do not mix spectra of one molecule across cohorts.

### 3.6 Preprocessing rules
1. **Verify and audit.** Checksums, row counts, null counts (adduct, instrument, collision energy), peak-array alignment, finite values. Report what you observe.
2. **Clean each spectrum.** Drop non-finite or non-positive m/z; keep intensities non-negative and renormalise to a base peak of 1. Ignore peaks above the precursor by more than 1.5 Da when building fragment features.
3. **Adduct-aware neutral mass.** Parse the adduct (`[nM + delta]z`), invert `m/z = (n*M + delta - z*m_e) / |z|` (`z` is the **signed** charge, negative for negative ions, `m_e` the electron mass) and obtain the neutral monoisotopic mass of each spectrum. Combine the spectra of a molecule into a **consensus neutral mass**: sort the estimates, take the largest cluster within 20 ppm (plus 0.003 Da), use its median, and break ties toward the smaller mass. The result must not depend on spectrum order and must never be NaN when estimates disagree.
4. **Candidate pool.** Keep catalog structures with `exact_mass` within a tolerance of the consensus mass, capped at 600 nearest by mass (many structures share almost the same mass, so break ties deterministically with a stable sort, then SMILES). Choose the tolerance **on training molecules only** (rule of thumb: the 97th percentile of the training absolute ppm error, never below 5 ppm). The ppm-error distribution can have a heavy tail: print it, and report the resulting pool sizes and pool recall, including whether the cap rather than the tolerance limits the pool.
5. **Order-invariant aggregation.** Any per-spectrum quantity becomes a per-molecule quantity with a symmetric function (mean, median, 90th percentile, count). Test it: shuffling spectra must give an identical ranking.
6. **Fit on training only:** imputation, clipping bounds, scalers, bit selection, percentile rules. Never on validation or final-test data.
7. **Missing values.** Decide explicitly how missing collision energy and instrument type are treated and say why.
8. Save the configuration, seed and the IDs of every cohort so a reviewer can reproduce your run.

### 3.7 Candidate catalog
`structure_catalog.parquet` has one row per distinct tautomer-canonical identity, sorted by `exact_mass`. Compute Morgan fingerprints (radius 2, 2048 bits) from `smiles` (snippet in `data/README.md`) and keep them aligned to the row index. It contains the structures of all 650 selected molecules among about 274,000 others and carries **no split information**; do not try to infer from the catalog which structures are held out, and do not remove any structure from it.

### 3.8 Small quantum cohort (because QPU time is limited)
Quantum models cannot afford the full data. Draw a deterministic, label-free cohort from your frozen classical results:

| Item | Recommended size | Definition |
|---|---|---|
| Frozen queue | all 100 internal-test molecules | Round-robin over the `stratum` column of `identity_manifest.parquet`: sort by `(stratum, sha256(f"{SEED}:{molecule_id}"))`, number each molecule within its stratum, then sort by `(that number, hash)`. Uses observable metadata and a hash only, no labels and no model output. |
| Quantum test molecules `U` | first 40 queue molecules that have at least `K` candidates in their pool (skip and report any others) | Same molecules for every method |
| Shortlist `K` | 2 | Top-K candidates by frozen M1 score (ties by SMILES). Gives `U*K` test pairs. |
| Quantum train anchors | from training molecules | For each training molecule (hash order) whose truth is in the pool: one **positive** pair (true candidate) and one **hardest negative** pair (highest M1 score among wrong candidates; prefer out-of-fold scores). |
| M2 anchors `A` | 12 pairs from 12 distinct molecules, alternating positive and negative (6 and 6) | Kernel training set |
| VQC training pairs `T` | 16 pairs from 8 further molecules (disjoint from the anchors) | Hardware-in-the-loop training |
| QAOA statistics | up to 30,000 training pairs | Relevance and redundancy of the 4 quantum features |

Sizes may go **down** if your quota requires, never up silently. Save the chosen molecule IDs. Quantum scores may only **re-order** the K shortlisted candidates of each molecule; ranks beyond K keep the M1 order. The best possible quantum MRR is therefore bounded by how often the truth is inside the shortlist: report it.

**The four quantum pair features** (computed for every molecule-candidate pair, with no label access):

| Feature | Meaning |
|---|---|
| `ppm_rank_pct` | Percentile rank of the candidate's median absolute ppm mass error within its pool |
| `ll_rank_pct` | Percentile rank of the candidate's fingerprint log-likelihood within its pool |
| `wtani_minus_max` | Probability-weighted Tanimoto minus the best value in the pool |
| `ll_minus_max` | Fingerprint log-likelihood minus the best value in the pool |

For **training** pairs, compute the fingerprint-based features from the **out-of-fold** fingerprint probabilities of Section 4.1 (never from a model that saw the molecule). Percentile ranks are pandas-style (`rank(pct=True)`, average ties). Fit the scaler on **all** training pairs (not only the anchors): 2nd and 98th percentile per feature, clip, map to an angle in `[0, pi]`, and record how many test values are clipped.

---

## 4. Methods to implement and compare

| # | Method | Type | Open-source stack |
|---|---|---|---|
| **M1** | **Fingerprint estimator plus LambdaRank ranker** | Classical ML | `rdkit`, `lightgbm` (or `xgboost`), `scikit-learn` |
| **M2** | **Quantum-kernel SVM (QSVC)** | Hybrid, kernel | `qiskit`, `qiskit-ibm-runtime`, `sklearn.svm.SVC(kernel="precomputed")` |
| **M3** | **QAOA feature selection, then VQC** | Hybrid, variational | `qiskit`, `qiskit-ibm-runtime`, a derivative-free optimiser written by you |

M1, M2 and M3 are mandatory. You may add extra baselines.

### 4.1 M1: classical fingerprint and ranking pipeline
1. **Per-spectrum descriptor** (no label information; reference design, 216 values): 100 fragment-m/z bins of 10 Da up to 1000 Da; 80 neutral-loss bins of 5 Da up to 400 Da (neutral loss = precursor m/z minus fragment m/z); 12 diagnostic neutral-loss sums (H2O, NH3, CO, CO2, HCOOH, CH3, CH4O, C2H4, 2xH2O, SO2, C3H6, HF; tolerance 0.01 Da); adduct class one-hot (8 classes); top-5 peak m/z; scalars (neutral mass, precursor m/z, number of peaks, spectral entropy, maximum m/z, precursor m/z modulo 1, polarity flag, collision energy, summed square-root intensity, peaks above 10 % intensity, mean m/z). Weight peaks by the square root of intensity; normalise the two binned blocks to sum 1. Fragments above 1000 Da fall into the last bin; neutral losses outside 0.5 to 400 Da are ignored.
2. **Fingerprint targets.** Compute the Morgan fingerprint of each training molecule's true structure. Select **96 stable bits** (prevalence between 5 % and 95 % in training molecules, closest to 50 %).
3. **Fingerprint estimator.** One small LightGBM classifier per selected bit (shallow trees, about 120 rounds, subsampling), trained on **spectrum rows**, grouped by molecule. Create **out-of-fold** bit probabilities with 5 group folds for all training molecules; use them (not in-sample predictions) when you build training pairs for the ranker.
4. **Molecule-level evidence.** Average the bit probabilities over the molecule's spectra (and average the 12 diagnostic losses).
5. **Pair features** for every (molecule, candidate in pool) pair:

| Group | Features |
|---|---|
| Fingerprint agreement | Mean Bernoulli log-likelihood of the candidate's bits under the predicted probabilities (clip probabilities to `[1e-3, 1-1e-3]`); probability-weighted Tanimoto; pool percentile ranks and differences to the pool maximum |
| Mass agreement | Median and maximum absolute ppm error across spectra between observed precursor and the candidate's expected adduct m/z; fraction of spectra within tolerance; ppm percentile rank |
| Context | Pool size, number of spectra |
| Candidate descriptors | The 20 descriptors from the catalog (`c_` prefix) |
| Fragment evidence | The 12 molecule-level diagnostic neutral-loss sums, plus six interactions with the matching functional-group counts (H2O with OH, NH3 with amine, CO2 and HCOOH with COOH, CO with carbonyl, CH3 with ether) |

6. **Ranker.** `LGBMRanker(objective="lambdarank")` with one query group per training molecule; skip training molecules whose truth is not in the pool and report how many. Select among a small pre-declared grid (for example 3 settings) by 5-fold **group** cross-validation on MRR@25.
7. **Inference** (`rank_molecule`): all spectra in, consensus mass, pool, pair features, ranker score, top-25 unique SMILES out. No label access. Time it per molecule.
8. **Controls and ablations** (all on identical molecules and pools): random order (seeded); mass only (smallest median ppm first); fingerprint likelihood only; M1 without spectral features (mass plus candidate descriptors); M1 without fragment features; **one representative spectrum versus all spectra**; shuffled-spectra order invariance (identical rankings required).
9. **Reporting:** on the internal 100 (labelled), report MRR@25, Top-1, Hit@5, Hit@25, pool recall, median pool size, mean Tanimoto of the top-1 candidate to the truth (scoring only), and p50/p95 latency per molecule. Add a short sensitivity note: how does pool recall react to the tolerance? Then freeze M1 and write the final-150 predictions (you cannot score them yourself). Mark any change made after seeing internal-100 scores as **post-hoc and exploratory**.

### 4.2 M2: QSVC on hardware
Task for each shortlisted pair: classify "true candidate" (1) versus "wrong candidate" (0) from the 4 scaled angles; the SVM decision score re-orders the K candidates.

1. **Measurement-free feature map `U(x)`** on 4 qubits (one per feature), features as angles in `[0, pi]`.
2. **Overlap (compute-uncompute) circuit** `U(x_i)` followed by `U(x_j)^dagger`, measure all qubits. The kernel entry is `K_ij = count("0000") / shots`, **measured on the QPU**. Save the standard error `sqrt(p(1-p)/shots)`.
3. Measure the **anchor Gram matrix** (upper triangle, symmetric) and the **test-vs-anchor cross matrix**. Batch circuits into as few jobs as possible (a few hundred circuits per job) and save the row/column mapping. The Gram diagonal may be set to 1 by definition; state that it was **not** measured.
4. Check symmetry and eigenvalues. If the measured Gram is not positive semidefinite, clip negative eigenvalues of the **training** Gram only, with a documented, consistent treatment of the cross matrix. Never use test labels.
5. Fit `SVC(kernel="precomputed", C=1.0)` on the anchors and score the cross matrix. SVM decision scores are not probabilities.
6. **Required comparison:** a **basic** map versus an **improved** map (Section 5.1), both measured on the same backend. **Control:** a classical RBF SVC (`gamma="scale"`) on the identical anchors and test pairs.
7. Re-rank each molecule's K shortlisted candidates by descending score (ties by M1 rank), keep ranks beyond K from M1.

### 4.3 M3: QAOA, then VQC
**QAOA feature selection (4 qubits, one per quantum feature).**
1. On training pairs compute **relevance** `r_i` (`sklearn.feature_selection.mutual_info_classif` with `random_state=SEED` on up to 30,000 sampled out-of-fold pairs, normalised by its maximum) and **redundancy** `red_ij` (absolute Spearman correlation, zero diagonal).
2. Build a QUBO that picks exactly `k = 2` of 4 features:
   `E(z) = -sum_i r_i z_i + alpha * sum_{i<j} red_ij z_i z_j + lambda * (sum_i z_i - k)^2`,
   with `alpha = 0.5`, `lambda = 1.0` (log your choice). In matrix form `E(z) = z^T Q z` with `Q_ii = -r_i + lambda(1 - 2k)` and `Q_ij = alpha red_ij + 2 lambda` for `i < j`; this drops the constant `lambda k^2`, so state which energy convention you report.
3. Convert to Ising coefficients analytically with `z = (1 - s)/2`: `h_i = -Q_ii/2 - (1/4) sum_{j != i} Q_ij`, `J_ij = Q_ij/4`. The Ising energy equals the QUBO energy up to an additive constant (`sum_i Q_ii/2 + sum_{i<j} Q_ij/4`); verify numerically on all 16 bitstrings that the two differ by that constant only.
4. Depth-1 QAOA circuit: Hadamard on every qubit, `rz(2 gamma h_i)`, `rzz(2 gamma J_ij)`, `rx(2 beta)` on every qubit, measure all. **State your bit-order convention:** Qiskit counts are little-endian, so qubit `i` is the character at position `n - 1 - i`.
5. Run a grid of `(gamma, beta)` pairs on hardware at 1024 shots, one batched job per round (reference design: a 3 x 3 grid over `gamma` in {0.4, 1.0, 1.6} and `beta` in {0.3, 0.8, 1.3}, then two refinement rounds of the 8 neighbours of the best point with step 0.3 and then 0.15). From measured counts compute the mean QUBO energy, the fraction of **feasible** bitstrings (exactly 2 ones) and the fraction equal to the exact optimum. Choose `(gamma, beta)` by lowest measured mean energy, then select the **most frequent feasible mask**.
6. Brute-force the exact optimum on CPU as a **labelled reference only** and report the gap, the feasibility rate and the energy of a uniformly random bitstring.

**VQC on hardware.**
1. **BEFORE:** all 4 quantum features, one qubit each; **AFTER:** the 2 QAOA-selected features (Section 5.1).
2. Angle-encode features with `ry`, add trainable `ry` layers, each followed by the `cx` cascade `cx(i+1, i)` for `i = n-2, ..., 0` (so every feature influences qubit 0), and measure **qubit 0** only. Class probability = share of shots reading `1`. Fix and document this readout convention.
3. Train with a **hardware-in-the-loop derivative-free optimiser** (reference design: a batched (1+8) evolution strategy, 4 generations, step size 1.0 shrinking by 0.6 per generation, 256 shots; SPSA is an acceptable alternative). Start from a centre drawn from `N(0, 0.5^2)`; in each generation evaluate the current best-so-far centre plus 8 Gaussian perturbations (standard deviation = step size), keep the best (elitist), and move the centre there. Each generation is **one batched job** containing every candidate parameter vector on every training pair.
4. Loss: mean binary cross-entropy on measured probabilities clipped to `[0.02, 0.98]`, plus `0.01 * ||w||^2`. **Every loss value must come from QPU measurements.** Log loss versus number of QPU calls.
5. Inference at 512 shots on the identical test pairs; re-rank the K shortlisted candidates by descending probability (ties by M1 rank).
6. Do not use Qiskit Machine Learning's default `VQC` or `SamplerQNN` unless you can prove its sampler is a real-hardware one.

### 4.4 Recommended configuration
Keep these in one `config` cell at the top of the notebook. Deviate only **downward** and disclose every change and why.

| Parameter | Value | Meaning |
|---|---|---|
| `SEED` | 2026 | Global seed (selection hashes, folds, optimiser, bootstrap) |
| Cohorts | 400 / 100 / 150 | Train / internal (labelled) / final (unlabelled, organiser-scored) molecules |
| `TOP_K` | 25 | Length of the ranked list |
| Pool cap | 600 | Maximum candidates per molecule |
| `FP_BITS_SELECTED` | 96 | Fingerprint bits modelled by M1 |
| `K_SHORTLIST` | 2 | Candidates re-ranked by quantum models |
| `U` | 40 | Quantum test molecules |
| `A` | 12 | QSVC anchors |
| `T` | 16 pairs (8 molecules) | VQC training pairs |
| Quantum features / qubits | 4 features; 4 qubits (QSVC, QAOA, BEFORE VQC), 2 qubits (AFTER VQC) | Hardware-friendly |
| Shots | 256 (kernels, VQC training), 512 (VQC inference), 1024 (QAOA) | Per circuit |
| `QAOA_K` | 2 of 4 | Features entering and leaving QAOA |
| VQC trainable angles | 4 (BEFORE), 4 (AFTER: 2 qubits x 2 layers) | Keeps circuits shallow |
| Bootstrap | 1000 resamples (classical), 2000 (paired quantum deltas) | Molecule-level intervals |
| `QPU_SOFT_BUDGET_SECONDS` | 420 | Planned usage; keep the rest as retry reserve |
| `QPU_SAFETY_RESERVE_SECONDS` | 100 | Never plan to spend this |
| `QPU_HARD_BUDGET_SECONDS` | min(remaining quota, 600) | Never exceed |
| `RUN_REAL_QPU` / `LIVE_JOB_ACK` | `False` by default | Set both to `True` deliberately before any submission |
| Transpilation | `generate_preset_pass_manager(optimization_level=1, backend=backend, seed_transpiler=SEED)` | ISA circuits for the chosen device |

Hardware access skeleton (adapt and test against your installed SDK; it shows the pattern, not a result):
```python
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.executor_sampler import Sampler   # client-side Sampler, qiskit-ibm-runtime >= 0.50
# older SDKs: from qiskit_ibm_runtime import SamplerV2 as Sampler

service = QiskitRuntimeService()                           # token from environment or saved account
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=4)
assert not getattr(backend, "simulator", False)            # simulators are prohibited
sampler = Sampler(mode=backend)
job = sampler.run(isa_circuits, shots=SHOTS)
save_job_id(job.job_id())                                  # persist BEFORE waiting
result = job.result()
counts = [dict(getattr(pub.data, next(iter(pub.data.keys()))).get_counts()) for pub in result]
usage = job.metrics().get("usage", {})                     # QPU seconds (key name depends on SDK version)
```
Write measurements to a **named classical register** and read counts from that register name. Every circuit must be fully bound and measured before submission.

### 4.5 Optional improvement round (use the remaining budget)
If your first round leaves budget, run a second, better-designed quantum round. It reframes the task as one decision per molecule: **should M1's top-2 order be kept or swapped?**

1. **Cross-fit M1 scores.** Refit the ranker with the same hyperparameters in 5 group folds over training molecules and obtain **out-of-fold** M1 scores for every training pair.
2. **Pairwise keep/swap task.** For each training molecule whose truth is in its out-of-fold top-2: feature vector = (rank-1 minus rank-2) of `[M1 score, ppm_rank_pct, ll_rank_pct, wtani_minus_max]`; label `keep = 1` if the rank-1 candidate is correct, else 0 (swap needed). Test molecules get the same differences from the frozen M1.
3. **Balanced anchors:** 8 keep and 8 swap (16 total, shuffled). Take the first 8 of each class in hash order. Fit the scaler (2nd to 98th percentile per feature, mapped to `[0, pi]`) on the difference vectors of **all** eligible training molecules, not only the anchors.
4. **Hardware-native map:** `line` map = `H`, `rz(x)` per qubit, a **nearest-neighbour CZ chain**, then `ry(x)`. Chain entanglers map directly onto heavy-hex hardware without SWAPs. Compare with the basic map.
5. **Hardware error mitigation:** also measure each anchor's and each test point's **self-overlap** `K(x, x)` (ideal 1.0) on the QPU and normalise: `K~(x, y) = K(x, y) / sqrt(K(x, x) K(y, y))`, clipped to `[0, 1]`. Report the raw and normalised results separately.
6. **Decision rule.** SVM with `C` in {0.1, 1, 10}; choose `C` and a keep-threshold by leave-one-out on the anchors, maximising `p_keep * TPR + (1 - p_keep) * TNR` where `p_keep` is the training keep rate (the always-keep rule scores exactly `p_keep`). Always-keep, which is identical to M1, is the default unless leave-one-out shows a gain. Anchor labels are train labels only.
7. **Controls on identical anchors and test pairs:** classical RBF-SVC (`gamma="scale"` or `1/(n_features * variance)`; state which), classical linear SVC, the **closed-form base kernel** `prod_i cos^2((x_i - y_i)/2)` as a clearly labelled **analytic reference** (not a quantum run), a larger-data classical reference that is explicitly labelled as not size-matched, and an **oracle ceiling** (perfect keep/swap, uses evaluation labels, reference only).
8. Report how far hardware kernels are from the closed form (for example RMSE and correlation, before and after normalisation) and whether mitigation changes the final ranking.

---

## 5. Circuit BEFORE / AFTER on real hardware

### 5.1 What to compare
Run a **BEFORE** (basic) and an **AFTER** (improved) circuit on the **same IBM backend, same quantum test molecules, same shots, same task.** The structure below is the reference design (a template, not numbers to match).

| Item | BEFORE (basic) | AFTER (improved) |
|---|---|---|
| QSVC feature map | Per qubit: `H` then `rz(x_i)` (4 qubits) | The basic map, then a **ring of `cz`** between neighbours (including last to first), then `ry(x_i)` per qubit |
| QSVC kernel circuit | `U(x_i)` then `U(x_j)^dagger`, measure all | Same overlap structure with the improved map |
| VQC input | All 4 quantum features, 4 qubits | The 2 **QAOA-selected** features, 2 qubits |
| VQC ansatz | `ry(x_i)` per qubit, one trainable `ry(w_i)` layer, `cx` cascade into qubit 0, measure qubit 0 | `ry(x_i)` per qubit, **two** trainable layers each followed by the `cx` cascade, measure qubit 0 |
| Readout | Class probability = share of shots with qubit 0 equal to 1 | Same |
| Optional round-2 map | n/a | `line` map (nearest-neighbour CZ chain) |

If features and circuit both change, label the result a **pipeline-level** comparison. To attribute the effect, add an **ablation** (the improved circuit on the original features, if budget allows). If AFTER is worse, keep the result and explain it.

### 5.2 Evidence required per comparison
Your notebook must **create both circuits in code** (functions taking your features and parameters, not pasted images) and, in cells that run **before** the job is submitted:
- [ ] Print the **logical** circuits as text (`circuit.draw("text")`) and save figures (`circuit.draw("mpl")`) for BEFORE and AFTER
- [ ] Print and draw the **transpiled** circuits for your backend, with the physical qubit layout visible
- [ ] Print a table with depth, size, two-qubit gate count, layout and circuit hash for BEFORE and AFTER
- [ ] Save the exact submitted circuits (`qpy.dump`) next to the job ID, so a judge can compare the diagram with what ran

Then provide: per-molecule measured scores and shot (binomial) uncertainty, the thresholded or re-ranked decisions, the per-molecule BEFORE-versus-AFTER change, and the resulting MRR@25 and Hit@K on the matched cohort with a paired bootstrap interval versus M1.

### 5.3 Working inside about 10 minutes of QPU time
- **QPU time is not wall-clock.** The 10 minutes count execution on the processor; queue time is extra. Plan for waiting.
- Cost is roughly a fixed amount per job plus a small amount per circuit (it grows with shots), so **batch many circuits into one job**. Check `job.metrics()` after each job and `service.usage()` before you start.
- Pick the backend with `service.least_busy(operational=True, simulator=False, min_num_qubits=4)` (or name one) and **assert it is not a simulator**. Use the **same backend for every stage** so results are comparable.
- Transpile with the preset pass manager for that backend (fixed `seed_transpiler`). **Choose physical qubits deliberately** (recommended): use calibration data or a short readout-health job and pass them as `initial_layout`; record the layout per job.
- **Save each job ID immediately** after submission. If the notebook restarts, retrieve with `service.job(job_id)` instead of resubmitting.
- Maintain an **append-only usage ledger** (job ID, stage, backend, shots, circuits, QPU seconds, molecule IDs, circuit hashes) and a pre-submission **budget guard** that refuses a batch if `planned + reserve > remaining`. Unknown usage is `None`, never 0.
- Validate every result: shots returned equal shots requested, one result per circuit, mapping to molecules preserved. Fail closed on cancelled or timed-out jobs. Stop with `BUDGET_EXHAUSTED` rather than overspend.

**Plan your circuit count before submitting** (A anchors, U molecules, K shortlist, F map variants, C QAOA points, G generations, L candidates per generation, T training pairs):

| Stage | Circuits |
|---|---|
| Calibration (tiny, train data only) | about 10 to 20 |
| QSVC per map | `A(A-1)/2` (Gram) `+ U*K*A` (cross) |
| QAOA | `C` (about 25 for the reference grid), one job per round |
| VQC training per variant | `G * L * T` (one job per generation) |
| VQC inference per variant | `U*K` |
| Optional round 2 per map | `A(A-1)/2 + A + U*A + U` (Gram, anchor self-overlaps, cross, test self-overlaps) |

Suggested plan: calibration, then QAOA, then QSVC (basic and improved), then VQC (BEFORE and AFTER), then the optional round, keeping at least 20 % in reserve. Measure the cost per circuit in the calibration job and **recompute your plan from it** before you scale up. If the plan does not fit, reduce `U` or `A` first.

### 5.4 Worked example: the BEFORE and AFTER circuits on real hardware
This example shows how to **build, print, transpile, save and submit** a BEFORE and an AFTER circuit on the same hardware, task and molecules. It contains no measured results: every number in it must come from your own run. The diagrams below are the logical circuits drawn with **symbolic parameters** (`a`, `b` are the scaled angles of two pairs, `x` the angles of one pair, `w` the trainable weights).

**Step 1: create both circuits in code** (functions that take your features and parameters, not pasted images). Write one builder for the QSVC feature map and kernel (overlap) circuit and one for the VQC, each with a BASE and an IMPROVED variant as described in Section 5.1, so that the same function produces the circuit you print, transpile, save and submit. Build them first with symbolic parameters for printing and diagramming, then bind real values before submission (`assign_parameters` or build with floats) so every submitted circuit is fully bound and measured.

**Step 2: print the logical circuits.** What `circuit.draw("text")` shows for the four designs (symbolic parameters):

QSVC kernel, BEFORE (basic map, then the inverse map of the second point, measure all):
```text
     ┌───┐┌──────────┐┌───────────┐┌───┐┌─┐
q_0: ┤ H ├┤ Rz(a[0]) ├┤ Rz(-b[0]) ├┤ H ├┤M├─────────
     ├───┤├──────────┤├───────────┤├───┤└╥┘┌─┐
q_1: ┤ H ├┤ Rz(a[1]) ├┤ Rz(-b[1]) ├┤ H ├─╫─┤M├──────
     ├───┤├──────────┤├───────────┤├───┤ ║ └╥┘┌─┐
q_2: ┤ H ├┤ Rz(a[2]) ├┤ Rz(-b[2]) ├┤ H ├─╫──╫─┤M├───
     ├───┤├──────────┤├───────────┤├───┤ ║  ║ └╥┘┌─┐
q_3: ┤ H ├┤ Rz(a[3]) ├┤ Rz(-b[3]) ├┤ H ├─╫──╫──╫─┤M├
     └───┘└──────────┘└───────────┘└───┘ ║  ║  ║ └╥┘
c: 4/════════════════════════════════════╩══╩══╩══╩═
```
QSVC kernel, AFTER (adds the CZ ring and an `Ry(x)` layer, and the inverse of both): the same skeleton, with four `CZ` gates (0-1, 1-2, 2-3, 3-0) and four `Ry` gates inserted in each of the two halves. Print it yourself; the logical depth is clearly larger than BEFORE.

VQC, BEFORE (4 qubits, one trainable layer, 4 weights):
```text
     ┌──────────┐┌──────────┐          ┌───┐┌─┐
q_0: ┤ Ry(x[0]) ├┤ Ry(w[0]) ├──────────┤ X ├┤M├
     ├──────────┤├──────────┤     ┌───┐└─┬─┘└╥┘
q_1: ┤ Ry(x[1]) ├┤ Ry(w[1]) ├─────┤ X ├──■───╫─
     ├──────────┤├──────────┤┌───┐└─┬─┘      ║
q_2: ┤ Ry(x[2]) ├┤ Ry(w[2]) ├┤ X ├──■────────╫─
     ├──────────┤├──────────┤└─┬─┘           ║
q_3: ┤ Ry(x[3]) ├┤ Ry(w[3]) ├──■─────────────╫─
     └──────────┘└──────────┘                ║
c: 1/════════════════════════════════════════╩═
```
VQC, AFTER (2 QAOA-selected features, two trainable layers, 4 weights):
```text
     ┌──────────┐┌──────────┐┌───┐┌──────────┐┌───┐┌─┐
q_0: ┤ Ry(x[0]) ├┤ Ry(w[0]) ├┤ X ├┤ Ry(w[2]) ├┤ X ├┤M├
     ├──────────┤├──────────┤└─┬─┘├──────────┤└─┬─┘└╥┘
q_1: ┤ Ry(x[1]) ├┤ Ry(w[1]) ├──■──┤ Ry(w[3]) ├──■───╫─
     └──────────┘└──────────┘     └──────────┘      ║
c: 1/═══════════════════════════════════════════════╩═
```

| | BEFORE (basic) | AFTER (improved) |
|---|---|---|
| QSVC feature map | `H`, `Rz(x)` per qubit | Same, plus a ring of 4 `CZ` and a second `Ry(x)` layer |
| QSVC two-qubit gates (logical) | 0 | 8 (4 in `U(x_i)`, 4 in its inverse) |
| VQC features and qubits | 4 features, 4 qubits | 2 selected features, 2 qubits |
| VQC trainable layers and weights | 1 layer, 4 weights | 2 layers, 4 weights |
| VQC two-qubit gates (logical) | 3 `CX` | 2 `CX` |
| Readout | Qubit 0 only; P(class 1) = share of shots reading 1 | Same |

**Step 3: transpile for your real backend and print it.** Use the same backend, pass manager and (recommended) `initial_layout` for BEFORE and AFTER:
```python
from qiskit.transpiler import generate_preset_pass_manager
pm = generate_preset_pass_manager(optimization_level=1, backend=backend, seed_transpiler=SEED)  # optionally initial_layout=[...]
isa_before, isa_after = pm.run(bound_before), pm.run(bound_after)
for tag, c in {"BEFORE": isa_before, "AFTER": isa_after}.items():
    print(tag, "depth:", c.depth(), "size:", c.size(), "2q gates:", c.num_nonlocal_gates(), "ops:", dict(c.count_ops()))
    print(c.draw("text", fold=200)); c.draw("mpl", filename=f"figures/{tag.lower()}_isa.png")
```
Expect gates to be rewritten into the device's native set (for IBM Heron/Eagle processors this is typically `Rz`, `sx`, `x` plus `cz` or `ecr`), `H` and `Ry` to expand into several native gates, the `cz` ring to need routing on a sparse coupling map, and the diagram to show the physical qubit each logical qubit was mapped to. Record the layout, depth and two-qubit count; they are part of your cost analysis and may explain differences between BEFORE and AFTER.

**Step 4: save exactly what you submit, then run both on the same hardware.**
```python
from qiskit import qpy
with open(f"evidence/{tag}_submitted.qpy", "wb") as f:
    qpy.dump(list_of_isa_circuits, f)          # next to the job ID
job = sampler.run(isa_circuits, shots=SHOTS)   # same backend, same molecules, same shots for BEFORE and AFTER
save_job_id(job.job_id())                      # before waiting
```

**Step 5: fill in your own comparison table** (this is the table judges expect; leave nothing blank and keep it identical in shape for BEFORE and AFTER):

| Item (measured in your run) | BEFORE | AFTER |
|---|---|---|
| Backend and physical qubits (layout) | | |
| Transpiled depth and two-qubit gate count | | |
| Shots per circuit, number of circuits, job ID(s) | | |
| QPU seconds charged | | |
| Per-molecule measured score and shot standard error | | |
| MRR@25 on the matched cohort with paired bootstrap interval versus M1 | | |
| Correct keep/order decisions when the truth is in the shortlist | | |
| Verdict (improved, no better, worse, not resolvable) and why | | |

**Step 6: interpret only what you measured.** If features and circuit both changed, call it a pipeline-level comparison and, budget permitting, add the ablation (improved circuit on the original features). If AFTER is not better, report it as measured and explain the likely causes (noise from the extra two-qubit gates, shorter or longer transpiled depth, fewer features, too little training, small cohort).

---

## 6. Benchmark metrics (report all that apply)

| Metric | Definition / note |
|---|---|
| **MRR@25** | Mean of `1 / rank` of the first correct structure; 0 if absent. Primary. |
| Top-1, Hit@5, Hit@25 | Share of molecules with the truth at rank 1, within 5, within 25 |
| Pool recall | Share of molecules whose truth is in the candidate pool (upper bound) |
| Shortlist recall | Share of quantum-cohort molecules whose truth is in the top-K (upper bound for quantum models) |
| Oracle ceiling | Best MRR a perfect keep/swap decision could reach (uses labels, reference only) |
| Top-1 Tanimoto | Mean Tanimoto of the top-1 candidate to the truth, scoring only |
| Bootstrap 95 % interval | Resample **molecules** (1000 for classical, 2000 for paired deltas); report for every headline metric |
| Paired delta versus M1 | Same molecules, `MRR_method - MRR_M1`, with interval; the only valid basis for a quantum-versus-classical statement |
| Correct decisions given truth in top-K | For re-rankers: how often the order was right when the truth was available |
| Kernel diagnostics | Symmetry, eigenvalues, mean off-diagonal, self-overlap, shot standard error; distance to the closed-form kernel if used |
| QAOA diagnostics | Mean measured QUBO energy, feasibility rate, gap to the exact optimum, versus a random bitstring |
| Latency | CPU p50/p95 per molecule for M1; QPU wall-clock per job separately for M2/M3 |
| Cost | QPU seconds per stage, number of circuits and jobs, transpiled depth and two-qubit gates |

Rules for honest comparison:
- Report classical results (150 molecules) and quantum results (about 40 molecules) in **separate tables**. Never compare them directly.
- Compare quantum methods **with M1 and the controls on exactly the same molecules**.
- With roughly 40 molecules, intervals are wide: write `NOT_STATISTICALLY_RESOLVABLE` when an interval includes zero change.

**Choosing the best method.** State your rule before looking at the final results: first eligibility (real-hardware provenance, no leakage, frozen before scoring), then highest MRR@25 on matched molecules, then cost and latency. If no method is distinguishable from M1, say the comparison is **inconclusive** and name an exploratory winner. Never imply quantum advantage without evidence.

---

## 7. Suggested notebook structure

You build this notebook. Every **code cell** must be preceded by a Markdown cell with six fields: **Purpose | Inputs | Method (with the maths) | Expected output | Interpretation | Failure mode**. Every **output** must be followed by an **Insights** cell written from the numbers actually printed. Outputs must be saved in the notebook (do not clear them).

| Section | Content | Done when |
|---|---|---|
| 00 Cover | Problem, data, allowed compute, status flags | Contract stated |
| 01 Environment | Package versions, seed, no-simulator scan of all notebook and source cells | Versions printed; scan passes |
| 02 Data integrity | Checksums, files, licence, row counts | Hashes logged |
| 03 Schema and audit | Columns, types, nulls, peak-array checks, join key checks | Audit printed |
| 04 EDA | Spectra per molecule, polarity, adducts, instruments, precursor mass, peak counts, collision-energy units, cohort composition | Figures saved |
| 05 Leakage register | Every column grouped with reasons | Excluded columns listed |
| 06 Chemistry utilities | Adduct parser, neutral mass, consensus mass, tautomer identity, unit tests | Checks pass |
| 07 Candidate pools | Tolerance from training, pool sizes, pool recall | Statistics printed |
| 08 Aggregation | Order-invariant molecule features, shuffled-order test | Identical rankings |
| 09 M1 training | Bits, OOF probabilities, pair table, ranker grid | Training record printed |
| 10 M1 benchmark | Internal 100, controls, ablations, tolerance sensitivity | Tables and figures |
| 11 Freeze | Model hash, catalog hash, config, quantum queue | Manifest written **before** the final-150 predictions are generated |
| 12 Failure analysis and final predictions | Internal-100 failure decomposition (catalog, pool, ranking) with intervals; top-25 predictions for the final 150 from the frozen M1, saved and hashed | Predictions file written; no scoring |
| 13 Quantum cohort | Queue, shortlists, anchors, scaler, shortlist recall | IDs saved |
| 14 Hardware access | Authenticate, real backend, quota, layout, budget guard | Backend is not a simulator |
| 15 Circuits before submit | BEFORE and AFTER circuits created, printed, transpiled, saved (QPY), metrics table | Diagrams and table shown |
| 16 Calibration | Tiny train-only job, cost per circuit, plan recomputed | Plan fits budget |
| 17 QAOA | QUBO, Ising check, hardware sweep, selected mask, exact reference | Feasibility and gap reported |
| 18 M2 QSVC | Gram and cross kernels on hardware, PSD check, SVC fit, control | Heatmaps, eigenvalues, scores |
| 19 M3 VQC | Hardware training, BEFORE and AFTER inference, ablation if budget allows | Loss curve, paired predictions |
| 20 Optional round | Pairwise keep/swap, normalised kernels, controls | Second comparison |
| 21 Job recovery | Re-query every job ID from IBM | All complete; shots match; not simulators |
| 22 Matched benchmark | All methods, controls and oracle on identical molecules, paired deltas | Table and charts |
| 23 Single-molecule demo | One molecule, all spectra, M1/M2/M3 top-25, annotated spectrum, structures | Status labelled measured versus cached |
| 24 Latency and cost | CPU versus QPU, cost per molecule | p50/p95 and QPU seconds shown |
| 25 Findings | Chosen method, claims, threats to validity, next steps | Every claim has a number |
| 26 Scorecard | Standards from Section 8 with evidence cell references | All rows filled |

Required figures: spectra-per-molecule and precursor-mass distributions, adduct and instrument mix, cohort composition, pool size and pool recall, MRR@25 and Hit@K with intervals for M1 and controls, QUBO and QAOA energy sweep with selected mask, kernel heatmaps (basic versus improved), VQC loss versus QPU calls, BEFORE/AFTER circuit diagrams (logical and transpiled), measured count or probability distributions, per-molecule paired changes, matched-cohort MRR with intervals, QPU usage versus jobs, an annotated example spectrum with its top candidates drawn as structures.

**Single-molecule function contract (section 23).** Provide one function, for example `identify_molecule(spectra, molecule_id, methods=("M1","M2","M3"), top_k=25)`, that takes **all spectra of one molecule** (query-input columns only; it must reject label columns) and returns, per method, up to 25 unique SMILES, status, number of input spectra, adducts and latency. For M2 and M3, only molecules actually measured on the QPU may be answered, and results read back from saved real-hardware jobs must be labelled `CACHED_REAL_QPU` with their job IDs. Otherwise return `NOT_MEASURED_ON_QPU` or `BLOCKED_QPU`. Never simulate.

---

## 8. Submission and judging checklist

Submit as described in `CONTRIBUTING.md` (fork, `submissions/GITHUB_USERNAME_NAME_FALL_FEST_2026/`, pull request). Judges score each standard against evidence in your executed notebook.

| # | Standard | Evidence judges look for |
|---|---|---|
| 1 | Clear problem statement and scope | Cover cell restating task, data, rules, limits, status flags |
| 2 | Clear outcome and deliverables | Predictions CSVs, evidence files, findings, single-molecule demo |
| 3 | Data use is correct and leakage-safe | Checksums, schema audit, feature register, molecule-level split, freeze before final predictions, pool and order-invariance checks |
| 4 | Methodology comparison | M1, M2, M3 plus controls on identical molecules; metrics table with intervals; stated selection rule |
| 5 | Real-hardware circuits, BEFORE/AFTER, within the QPU budget | Printed logical and transpiled circuits, depth and two-qubit tables, QPY files, job IDs, QPU seconds, non-simulator backend |
| 6 | Technical explanation and insights | Six-field Markdown before every code cell, Insights after every output, saved figures |
| 7 | Final comparison and end goals | Scorecard of accuracy, bound, cost, latency and risk, with paired deltas |
| 8 | Integrity and reproducibility | No simulator symbols, no secrets, every quantum number traceable to a job ID, notebook runs top to bottom, negative results reported |

Final checklist before you open your pull request:
- [ ] Notebook executed top to bottom with saved outputs
- [ ] Final-150 predictions written only after the freeze manifest, from the frozen models, and included as `predictions_final_test_150_M1.csv`
- [ ] All circuits created in code, printed, transpiled and saved (QPY) before submission to hardware
- [ ] Every job ID re-queried and confirmed non-simulator, with shots matching
- [ ] No API key, token or CRN anywhere in the repository
- [ ] Classical and quantum results reported in separate tables with intervals
- [ ] Limitations and negative results stated plainly

---

## 9. Licence and acknowledgements

The data derive from a public tandem mass spectrometry collection licensed **CC BY-NC 4.0**: non-commercial use only. They are provided for this hackathon; do not redistribute them outside the event. Cite the IBM Quantum Platform in your report. The notebook and code you write are your own work.
