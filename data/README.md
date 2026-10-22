# Data Card: Q-MetaboX LC-MS/MS Molecule-Identification Cohort
### IBM Fall Fest 2026 | Trinexis x IEEE | Domain: Healthcare / Life Sciences

This folder contains **every file you are allowed to use** for the challenge. Do not download or use any other dataset, library spectra, pretrained model or structure database (see Section 8).

---

## 1. Source, licence and provenance

| Item | Value |
|---|---|
| Original source | A public small-molecule tandem mass spectrometry collection (about 2.54 million experimental MS/MS spectra, about 275,000 distinct structures) |
| Licence | **CC BY-NC 4.0** (non-commercial, attribution required). The data are provided here for this hackathon only. Do not use them commercially or redistribute them outside the event. |
| Attribution | Credit to the original data providers is given by the organisers; do not remove this licence notice from any copy. |
| Nature of data | Experimental liquid-chromatography tandem mass spectra (LC-MS/MS) of small molecules, aggregated from several source libraries. No patient data, no personal data. |
| What we did | Selected **650 molecules** (all of each molecule's valid spectra) with a fixed seed, split them by molecule into train / internal test / final test, built a **structure-only candidate catalog**, replaced all identifiers by random ones, and separated the final-test inputs from the final-test labels. |

How the 650 molecules were chosen (for your methods section; you do not need to repeat it):
1. Keep spectra with at least 5 peaks and a precursor m/z between 50 and 2000; keep molecules with 2 to 60 such spectra.
2. Group by molecule (first 14 characters of the InChIKey) and stratify by median precursor-mass quartile, ionisation polarity and instrument group (40 strata).
3. Deterministic seeded round-robin over strata, then validate every structure with RDKit (valid SMILES, one tautomer-canonical identity per molecule, no collisions).
4. Deal the 650 molecules into train (400), internal test (100) and final test (150) within strata. **The split is by molecule: no molecule appears in more than one split.**

---

## 2. Files

| File | Rows | Molecules | Role | Labels inside? |
|---|---|---|---|---|
| `train_400.parquet` | 3,423 spectra | 400 | Fit every model (feature scaling, fingerprint models, rankers, quantum anchors) | Yes |
| `internal_test_100.parquet` | 1,127 spectra | 100 | Development and model selection (your validation set) | Yes |
| `final_test_150_inputs.parquet` | 1,620 spectra | 150 | **Final test: query spectra only. No answers are provided; the organisers score your predictions.** | **No** |
| `structure_catalog.parquet` | 274,195 structures | n/a | Candidate bank: every structure your ranker may propose | n/a (structures only) |
| `identity_manifest.parquet` | 650 molecules | 650 | Observable per-molecule metadata and strata (no labels) | No |
| `SHA256SUMS.txt` | n/a | n/a | Checksums of the files above | n/a |

Verify integrity first:
```bash
cd data && sha256sum -c SHA256SUMS.txt
```
Log the checksums in your notebook. If a checksum fails, re-download the file and do not continue.

---

## 3. Column dictionary

### 3.1 Query-input columns (present in all three spectrum files)
These twelve columns are what a real analyst has for an unknown molecule. The ten spectrum-describing columns (all except the two identifiers) are **the only columns you may use as model inputs**; the identifiers are for grouping only.

| Column | Type | Meaning |
|---|---|---|
| `molecule_id` | string | Random identifier of the molecule. All spectra with the same `molecule_id` come from the same molecule. Meaningless in itself. |
| `spectrum_id` | string | Random identifier of one spectrum. |
| `ms2_mzs` | list of float | Fragment m/z values of the MS/MS spectrum. |
| `ms2_normalized_intensities` | list of float | Fragment intensities, base peak scaled to 1. Same length as `ms2_mzs`. |
| `base_peak_intensity` | float | Raw intensity of the base peak. |
| `adduct` | string | Ion type, for example `[M+H]+`, `[M-H]-`, `[M+Na]+`, `[2M-H]-`. Needed to convert precursor m/z into a neutral mass. |
| `ionization_mode` | string | `positive` or `negative`. |
| `instrument_type` | string | Instrument family (free text, some missing). |
| `precursor_mz` | float | Measured m/z of the precursor ion. |
| `collision_energy_ev` | list of float or null | Collision energy converted to eV where possible. Often empty. |
| `collision_energy_orig` | string or null | Collision energy as originally reported. |
| `collision_energy_orig_units` | string | `eV`, `NCE`, `V` or `unknown`. Do not treat NCE or V as eV. |

### 3.2 Label and audit columns (only in `train_400` and `internal_test_100`)
**Never use these as model inputs on the final test.** They exist for training targets, grouping and auditing.

| Column | Meaning |
|---|---|
| `normalized_smiles` | Structure of the molecule (training target). |
| `inchikey`, `inchikey14` | InChIKey of the structure and its first 14 characters (connectivity layer). |
| `canonical_identity` | RDKit tautomer-canonical InChIKey14. **This is the identity used for scoring.** |
| `molecular_formula` | Formula of the structure. |
| `ingest_lib` | Source library of the spectrum (audit only). |
| `adduct_orig` | Adduct label before normalisation (audit only). |
| `precursor_error_ppm` | Reported precursor mass error computed with the true formula. It encodes the answer; never an input. |
| `num_peaks` | Number of peaks (derivable from the arrays). |
| `split` | `train` or `internal_test`. |

### 3.3 `structure_catalog.parquet` (candidate bank)
One row per distinct tautomer-canonical identity, sorted by `exact_mass` (the row position is your candidate index). It contains **no spectra, no split information and no link to any query**.

| Column group | Columns |
|---|---|
| Identity | `smiles`, `canonical_identity` |
| Mass and formula | `exact_mass` (neutral monoisotopic, Da), `formula` |
| RDKit descriptors | `n_heavy`, `n_rings`, `n_arom_rings`, `tpsa`, `logp`, `hbd`, `hba`, `rotb`, `n_N`, `n_O`, `n_S`, `n_hal`, `formal_charge`, `frac_sp3` |
| Functional-group counts | `n_OH`, `n_COOH`, `n_amine`, `n_carbonyl`, `n_ether`, `n_amide` |

The catalog does **not** ship fingerprints. Compute Morgan fingerprints (radius 2, 2048 bits) from `smiles` yourself; it takes minutes with RDKit:
```python
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
cat = pd.read_parquet("data/structure_catalog.parquet")
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fps = np.stack([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)).astype(np.uint8) for s in cat.smiles])  # row i = candidate i
```

### 3.4 `identity_manifest.parquet`
Observable metadata only: `molecule_id`, `split` (`train`, `internal_test`, `final_test`), `n_spectra`, `median_precursor_mz`, `polarity`, `instr_group` (`Orbitrap/FT`, `QTOF`, `timsTOF`, `other`, `unknown`), `n_adducts`, `mass_bin` (precursor-mass quartile 0 to 3) and `stratum` (`m<bin>|<polarity>|<instrument group>`). Use `stratum` to build the deterministic quantum-cohort queue (main README, Section 3.8).

---

## 4. How each file must be used

| Phase | Allowed files | Rule |
|---|---|---|
| Feature design, fingerprint models, ranker, scaler, quantum anchors, QUBO statistics | `train_400` (+ `structure_catalog`) | Fit on training molecules only. Use cross-fitting (group folds by molecule) wherever a train-side score is later used as a feature. |
| Hyperparameter choice, model selection, thresholds, shortlist size | Cross-validation inside `train_400` (group folds by molecule) | Choose settings here, then freeze them. |
| Evaluation | `internal_test_100` (labelled) | Report metrics here and draw the quantum cohort from it. Do not tune on it without labelling the change post-hoc. |
| Freeze | n/a | Save model hash, catalog hash, configuration and the quantum queue **before** generating the final-test predictions. |
| Final predictions | `final_test_150_inputs` | Write top-25 predictions for each `molecule_id` from the frozen models and submit them. The organisers score them. Do not retune afterwards; if you change anything, disclose it. |

Further rules:
- **A prediction is per molecule, not per spectrum.** For each `molecule_id`, use all its spectra and output one ranked list of at most 25 valid, unique SMILES. The result must not depend on the order of spectra.
- **Candidates come only from `structure_catalog`.** Do not prune it using split knowledge (for example, do not remove the structures of training molecules), so that all teams are comparable.
- **Score by identity:** two SMILES are the same molecule if their tautomer-canonical InChIKey14 match (RDKit `TautomerEnumerator().Canonicalize`, then `Chem.MolToInchiKey(mol)[:14]`).
- `final_test_150_inputs` must never be used for fitting, imputation, scaling or threshold selection (no transductive tricks).
- The quantum cohort is a small subset of the internal-test molecules (main README, Section 3.8). It is not a separate file.

---

## 5. Descriptive statistics (what the data look like)

| | Train | Internal test | Final test |
|---|---|---|---|
| Molecules | 400 | 100 | 150 |
| Spectra | 3,423 | 1,127 | 1,620 |
| Spectra per molecule (min / median / max) | 2 / 6 / 59 | 2 / 8 / 60 | 2 / 6 / 59 |
| Positive / negative spectra | 1,879 / 1,544 | 601 / 526 | 890 / 730 |
| Median peaks per spectrum | 31 | 29 | 29 |
| Precursor m/z (min / median / max) | 57 / 343 / 1505 | 109 / 353 / 1450 | 118 / 355 / 1034 |
| Distinct adduct labels | 29 | 22 | 20 |
| Molecules per instrument group (Orbitrap/FT, QTOF, timsTOF, other, unknown) | 80, 80, 81, 79, 80 | 20, 19, 23, 21, 17 | 32, 29, 31, 27, 31 |

Candidate catalog: 274,195 structures, neutral exact mass from about 31 to 3,400 Da (median about 333 Da).

Most frequent adducts: `[M+H]+` and `[M-H]-` by a wide margin, then (order varies by split) `[M+Na]+`, `[M+NH4]+`, `[M+CH2O2-H]-`, `[M-H2O+H]+` and dimer forms such as `[2M-H]-`.

---

## 6. Known data caveats (plan for these)

- **Several spectra per molecule** with different adducts, collision energies and instruments. Aggregate them in an order-invariant way.
- **Heterogeneous provenance.** Instrument type is missing for roughly 7 to 12 % of spectra, and `collision_energy_ev` is empty for roughly 35 to 46 % (collision energy is also reported in different units (`eV`, `NCE`, `V`, `unknown`)). Handle missing values explicitly and say how.
- **Spectra differ hugely in size**: from 5 peaks to more than 20,000 peaks. Peaks above the precursor m/z are usually noise, isotopes or adduct artefacts; decide how you treat them.
- **Adduct chemistry matters.** The neutral mass of a molecule comes from `precursor_mz` and `adduct`; every adduct in these files is parseable with a standard adduct formula table (`[M+H]+`, `[2M-H]-`, `[M+CH2O2-H]-`, and so on). Different spectra of one molecule can disagree, so use a robust consensus.
- **The catalog is large relative to the mass window.** Many catalog structures share (almost) the same mass, so mass alone does not identify a molecule.
- **The true structure may fall outside your candidate pool** (mass tolerance, odd adducts). Report this as a pool-recall limit; a ranker cannot recover what it never sees.
- **The selected molecules are not a random draw from the catalog.** They were chosen because they have 2 to 60 valid spectra, so they differ in size and polarity from the average catalog structure; the catalog is therefore not a neutral background and you should not exploit property differences as a shortcut.
- **Tautomers and stereochemistry.** Scoring uses tautomer-canonical InChIKey14, so stereo and tautomer differences do not count as errors.

---

## 7. Loading examples

```python
import pandas as pd
train  = pd.read_parquet("data/train_400.parquet")
inner  = pd.read_parquet("data/internal_test_100.parquet")
final_x = pd.read_parquet("data/final_test_150_inputs.parquet")     # inputs only
catalog = pd.read_parquet("data/structure_catalog.parquet")
manifest = pd.read_parquet("data/identity_manifest.parquet")

assert not ({"canonical_identity", "normalized_smiles", "inchikey14"} & set(final_x.columns))
```

---

## 8. Data rules (summary)

1. Use **only** the files in this folder. No external spectra, structures, fingerprints, embeddings, pretrained models or any other dataset.
2. Do not try to re-identify molecules or map identifiers back to the original source files. Identifiers were randomised on purpose.
3. Model inputs come from the ten spectrum-describing query columns and from the candidate catalog only (identifiers are for grouping only).
4. Never use `final_test_150_inputs` for fitting, scaling or tuning, and never try to recover its answers.
5. Keep the data inside this event; cite the source (Section 1) in your report.
