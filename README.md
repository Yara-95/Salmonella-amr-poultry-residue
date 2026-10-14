# Simulation-Based Framework for Poultry Antibiotic Residues and *Salmonella* Resistance

This repository contains the simulation code, scenario-level results, and figure
files accompanying the methodological proof-of-concept study:

**A Simulation-Based Framework for Integrating Poultry Antibiotic Residue Data
with Salmonella Resistance Profiles: A Methodological Proof-of-Concept**

## Important interpretation

All records and outcomes in this repository are synthetic. They do not represent
real poultry samples, *Salmonella* isolates, patients, or geographic regions.

The simulations evaluate how a machine-learning pipeline behaves when data are
generated under explicitly defined assumptions. They do **not** establish a
biological association between poultry antimicrobial residues and resistance in
human *Salmonella*, and they do not demonstrate clinical or surveillance
prediction. Any practical application would require validation with independent
biological data.

## Repository contents

- `simulation_scenarios.py` — Python script for generating and evaluating
  independent simulation scenarios.
- `scenario_results.csv` — scenario-level AUC-ROC summary results.
- `Figure8_scenario_robustness.jpg` — figure comparing model performance
  across simulation scenarios.
- Add any additional manuscript-related files here, using their exact filenames.

## Multi-scenario robustness analysis

The robustness analysis includes 14 scenarios (S0–S13). Each scenario is
evaluated using 30 independently generated datasets, with 2,500 records per
dataset. Each dataset is split into training and test sets using an 80/20
stratified split.

The scenarios vary assumptions about:

- The strength and form of the simulated residue–resistance relationship,
  including a true-null scenario with no predefined association.
- Whether MIC values are generated independently of, or conditionally on,
  the simulated resistance label.
- MIC overlap and MIC distribution.
- Resistant-class prevalence and class imbalance.
- Whether residue effects apply to all antibiotics or only selected antibiotics,
  and whether thresholds differ by antibiotic.

For each replicate, the script evaluates three feature sets:

1. **All features:** residue, MIC, antibiotic, and synthetic region.
2. **Without MIC:** residue, antibiotic, and synthetic region; this helps isolate
   the information associated with the simulated residue signal.
3. **MIC only:** MIC, antibiotic, and synthetic region; this helps show the
   contribution of MIC values generated conditionally on the simulated label.

Results are summarized using the mean test-set AUC-ROC and standard deviation
across independent replicates. The figure displays mean AUC-ROC ± SD; the
dashed line marks chance-level discrimination (AUC-ROC = 0.50).

## Interpretation of the null and MIC scenarios

- **S0 — True null:** The resistance label is independent of residue, and MIC
  is independent of the label. Performance near AUC-ROC 0.50 is expected when
  no association is encoded.
- **S1 — Residue null, MIC label-conditional:** The label is independent of
  residue, but MIC is generated conditional on the label. Performance from
  models including MIC therefore reflects the simulation design and must not
  be interpreted as evidence of a residue effect.
- In the remaining scenarios, the model can recover patterns that were
  deliberately encoded in the data-generating rules. Higher performance
  indicates recovery of those simulated rules, not discovery of a real-world
  biological relationship.

Because MIC is generated conditional on the simulated resistance label in
some scenarios, MIC can act as a proxy for that label by construction. The
simulated labels are not clinical susceptibility categories assigned using
CLSI/EUCAST breakpoints or epidemiological cut-off values.

## Requirements

The analysis script uses Python and the following packages:

- NumPy
- pandas
- scikit-learn

## Run the scenario analysis

From the repository directory, run:

```bash
python simulation_scenarios.py
```

By default, the script runs 30 replicates per scenario. To choose a different
number of replicates, provide it as a command-line argument:

```bash
python simulation_scenarios.py 30
```

The script writes scenario-level results to a CSV file. Check the output
filename in the script before running; it may differ from the committed
`scenario_results.csv`.

## Limitations

The scenario results depend on the specified data-generating rules, including
the residue range, resistance probabilities, MIC ranges, and class-conditional
MIC generation. These assumptions are methodological choices, not estimates
from linked field or clinical data.

The same residue range was applied across the four antibiotics for simulation
simplicity, despite drug-specific differences in residue distributions and
maximum residue limits. The available local field evidence primarily concerned
fluoroquinolone residues. The antibiotic-specific scenarios explore some
alternative assumptions but cannot replace drug-specific empirical data.

The results should therefore be interpreted only as a computational
proof-of-concept. Validation with independently collected and linked residue,
microbiological, and clinical data would be required before evaluating any
biological or practical application.

## Reproducibility note

The committed results and figure should be regenerated from the same version
of the analysis script and the stated replicate count. Verify that the
script-generated CSV filename matches the filename referenced above before
publishing or committing updated outputs.
