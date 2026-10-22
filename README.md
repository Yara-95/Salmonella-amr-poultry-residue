# Simulation-Based Framework for Poultry Antibiotic Residues and *Salmonella* Resistance

This repository contains the synthetic dataset, analysis code, figures, and result files accompanying the article:

Al-Hallak Y. *A simulation-based framework for integrating poultry antibiotic residue data with Salmonella resistance profiles: a methodological proof-of-concept.* BMC Veterinary Research (accepted; in production). DOI: [10.1186/s12917-026-06050-1](https://doi.org/10.1186/s12917-026-06050-1) (the DOI will resolve once the article is published online).

## Important interpretation

All observations, resistance labels, and geographic regions are synthetic. They do not represent real poultry samples, *Salmonella* isolates, patients, or geographic areas.

The simulations evaluate how a machine-learning pipeline behaves when data are generated under explicitly defined assumptions. They do not establish a biological association between poultry antimicrobial residues and resistance in human *Salmonella* isolates, and they do not demonstrate validated clinical or surveillance prediction. Any practical application would require validation with independent biological data.

## Repository structure

| Folder | Contents |
|---|---|
| `Dataset/` | `human_amr_poultry_residue_dataset_stm.csv` — the synthetic dataset (2,500 records) used for the main analysis |
| `Codes/` | `project_AMR_dataset.py` — main analysis (models, five-fold cross-validation, calibration, reference benchmark, SHAP)<br>`simulation_scenarios_v2.py` — multi-scenario robustness analysis (S0–S13)<br>`make_figure2.py` — redraws Figure 2 with panel labels A–D |
| `Results/` | CSV outputs reported in the article (see table below) |
| `Figures/` | Figures 1–8 of the article (image files) |

Figure 1 (MIC range overlap) is provided as an image only; the script that drew it is not included. The ranges it displays are those listed in Table 1 of the article.

### Result files and where they appear in the article

| File (`Results/`) | Content | Article |
|---|---|---|
| `model_comparison_summary.csv` | Test-set accuracy, AUC-ROC, AP for the three models | Table 2 |
| `cross_validation_summary.csv`, `cross_validation_fold_scores.csv` | Five-fold cross-validation results and fold-level scores | Table 3 |
| `reference_benchmark_summary.csv` | Reference classifier with knowledge of the generating rule | Results, "Comparison with the reference benchmark" |
| `calibration_summary_cv5.csv`, `calibration_curve_points.csv` | Brier scores and reliability-curve points (raw vs isotonic) | Figure 3 |
| `shap_global_verified.csv` | Global mean \|SHAP\| values | Figure 4 |
| `shap_cases_verified.csv`, `shap_additivity_check.csv` | Local SHAP values for records HUM-3201 and HUM-3921 and the additivity check | Figures 6–7 |
| `scenario_results_v2.csv` | Scenario-level results (S0–S13) | Table 6, Figure 8 |
| `software_versions.csv` | Python and package versions used for the primary analysis | — |

## Data-generating rule (base design)

1. Residue: *R* ~ Uniform(0.02, 5.0) mg/kg. Antibiotic (ampicillin, ciprofloxacin, tetracycline, streptomycin) and region (A–D) are assigned at random, independently of each other and of the outcome.
2. Label: *Y* ~ Bernoulli(p), with p = 0.80 if *R* > 1.25 mg/kg and p = 0.15 otherwise. The threshold and probabilities are design assumptions, not empirical estimates.
3. MIC: drawn from a uniform distribution within a range specific to the antibiotic and to the assigned label (Table 1 of the article).

## Multi-scenario robustness analysis

The analysis includes 14 scenarios (S0–S13). Each scenario is evaluated on 30 independently generated datasets of 2,500 observations. Within each replicate, the data are split 80/20 (stratified), and a Random Forest with 150 trees and maximum depth 12 is fitted and evaluated on the held-out set.

The scenarios vary:

- The strength of the simulated residue–resistance relationship, including no predefined association.
- The form of the relationship: step, logistic, or linear.
- Label-independent versus label-conditional MIC generation.
- MIC overlap and MIC distribution.
- Resistant-class prevalence, including 80% and 20% resistant scenarios (sampling without replacement from a larger simulated pool, so that no records are duplicated).
- Whether the residue effect applies to all antibiotics or only ciprofloxacin.
- Common versus antibiotic-specific residue thresholds.

Each replicate evaluates three feature sets with identical hyperparameters:

1. All features: residue, MIC, antibiotic, and synthetic region.
2. Without MIC: residue, antibiotic, and synthetic region (isolates the residue signal).
3. MIC plus categorical descriptors: MIC, antibiotic, and synthetic region, without residue (isolates the information carried by the label-conditional MIC).

### Contents of `scenario_results_v2.csv`

- `Resistant_pct`: **mean** percentage of resistant labels across the 30 replicates of each scenario.
- `AUC_*` and `SD_*`: mean test-set AUC-ROC and its standard deviation across replicates, for each feature set.
- `lo_*` and `hi_*`: 2.5th and 97.5th percentiles of AUC-ROC across replicates (all-feature and without-MIC models). These describe variability across simulated datasets; they are not confidence intervals for the mean.
- `Acc_all`: mean test-set accuracy of the all-feature model.

### Interpretation of the null scenarios

- **S0 (true null):** the label is independent of residue, and MIC is generated independently of the label. Performance near AUC-ROC 0.50 is expected when no association is encoded.
- **S1 (residue null, label-conditional MIC):** the label is independent of residue, but MIC is generated conditional on the label. Performance of models that include MIC reflects the MIC-generation design and must not be interpreted as evidence of a residue effect.

Higher model performance in the other scenarios indicates recovery of patterns deliberately encoded in the simulation. It does not demonstrate discovery of a real-world biological relationship.

## MIC generation and susceptibility interpretation

In the base design and most scenarios, MIC values are generated conditional on the simulated resistance label and antibiotic identity. MIC therefore carries information about the label by construction. Partially overlapping MIC ranges reduce perfect separation but do not remove this circularity.

The simulated labels are assigned by the data-generating rule. They are not susceptibility categories derived from CLSI/EUCAST clinical breakpoints or epidemiological cut-off values.

## Requirements

Python 3 with NumPy, pandas, scikit-learn, matplotlib, seaborn, SHAP, and SciPy:

```bash
python -m pip install numpy pandas scikit-learn matplotlib seaborn shap scipy
```

The package versions used for the primary analysis are recorded in `Results/software_versions.csv`. No pinned environment file is provided, so package versions may affect numerical results slightly.

## How to run

The scripts read and write files in the **current working directory**. Copy the dataset into the folder from which you run the scripts, for example:

```bash
cd Codes
cp ../Dataset/human_amr_poultry_residue_dataset_stm.csv .

# Primary analysis: models, cross-validation, calibration, reference benchmark, SHAP
python project_AMR_dataset.py

# Multi-scenario analysis (30 replicates per scenario by default)
python simulation_scenarios_v2.py
python simulation_scenarios_v2.py 30          # replicate count can be given explicitly

# Redraw Figure 8 from an existing scenario CSV without rerunning the simulations
python simulation_scenarios_v2.py --plot-only

# Redraw Figure 2 with panel labels A-D
python make_figure2.py
```

Output files are written to the working directory and overwrite existing files with the same names. The copies in `Results/` and `Figures/` are the outputs of the run reported in the article.

## Reproducibility

The scripts use fixed random seeds for dataset generation and train–test splitting. To compare with the reported results, use the same script version, scenario definitions, and replicate count, and record the Python and package versions used.

## Limitations

- The results depend on the specified data-generating rules, including residue ranges, resistance probabilities, MIC distributions, and label-conditional MIC generation. These are simulation assumptions, not estimates obtained from linked field or clinical data.
- The same residue range (0.02–5.0 mg/kg) and threshold are used across the four antibiotics for simulation simplicity. This does not imply that real residue distributions or maximum residue limits are equivalent across drugs.
- The available local field evidence primarily concerns fluoroquinolone residues. The antibiotic-specific scenarios explore selected alternative assumptions but cannot replace drug-specific empirical data.
- The multi-scenario analysis characterizes performance under controlled synthetic conditions. It does not establish generalizability to real biological populations.
- Validation using independently collected and appropriately linked residue, microbiological, and clinical data would be required before considering biological interpretation or practical application.

## License

All content in this repository (code, dataset, result files, and figures) is released under the
CC0 1.0 Universal public domain dedication (see LICENSE). If you use the code or data, citation
of the associated article is appreciated but not required.

## Citation

If you use the code or the synthetic dataset, please cite the article and the repository:

Al-Hallak Y. A simulation-based framework for integrating poultry antibiotic residue data with *Salmonella* resistance profiles: a methodological proof-of-concept. *BMC Veterinary Research*. 2026. DOI: 10.1186/s12917-026-06050-1.

Repository: https://github.com/Yara-95/Salmonella-amr-poultry-residue
