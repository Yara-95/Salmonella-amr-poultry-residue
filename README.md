Predicting Human Salmonella Antimicrobial Resistance from Poultry Antibiotic Residues
A Simulation-Based Machine Learning Proof-of-Concept with One Health Integration
________________________________________
Overview
This repository contains the simulated dataset, analysis code, and figures for the manuscript:
“Predicting Human Salmonella Antimicrobial Resistance from Poultry Antibiotic Residues: A Simulation-Based Machine Learning Proof-of-Concept with One Health Integration”
Submitted to BMC Veterinary Research (July 2026)
________________________________________
Background
Antimicrobial resistance (AMR) in foodborne pathogens such as Salmonella enterica represents a critical One Health challenge. Sub-therapeutic antibiotic residues in poultry meat are hypothesized to exert continuous selective pressure on bacterial populations. However, no publicly available dataset currently links measured poultry antibiotic residues to matched human Salmonella phenotypic resistance outcomes.
This study addresses this gap methodologically by constructing a controlled, rule-based synthetic dataset and testing whether an interpretable machine-learning pipeline can recover the encoded selective-pressure relationship.
________________________________________
Repository Contents
File	Description
human_amr_poultry_residue_dataset_stm.csv	Simulated dataset (2,500 records)
project_AMR.py	Full Python analysis pipeline
________________________________________
Dataset Description
The synthetic dataset includes 2,500 simulated human clinical cases with the following variables:
Variable	Type	Description
Poultry_Residue_Level_mg_kg	Continuous (0.02–5.0)	Poultry antibiotic residue level (mg/kg)
Antibiotic_Tested	Categorical	Ampicillin, Ciprofloxacin, Tetracycline, Streptomycin
Geographic_Location	Categorical	Region A, B, C, D
Human_MIC_mg_L	Continuous	Minimum inhibitory concentration (mg/L)
Human_Resistance_Outcome	Binary	1 = Resistant, 0 = Susceptible
Key design feature: MIC ranges for resistant and susceptible isolates were intentionally overlapped (~47–49%) to prevent trivial data leakage and force the model to rely on the residue-resistance relationship.
________________________________________
Methods
Machine Learning Pipeline
1.	Feature Engineering
–	One-hot encoding for categorical variables
–	Z-score standardization for Logistic Regression
2.	Models Compared
–	Baseline Random Forest (categorical features only)
–	Logistic Regression (all features, standardized)
–	Optimized Random Forest (all features, 150 trees, max_depth=12)
3.	Evaluation
–	80/20 stratified train-test split (random_state=7)
–	5-fold stratified cross-validation
–	Isotonic probability calibration
–	SHAP (TreeExplainer) for interpretability
Key Results
Model	Accuracy	AUC-ROC	Average Precision
RF Baseline	63.0%	0.471	0.628
Logistic Regression	77.8%	0.819	0.875
RF Optimized	79.6%	0.860	0.916
________________________________________
Requirements
•	Python 3.x
•	scikit-learn
•	SHAP
•	pandas
•	matplotlib
•	numpy
________________________________________
Usage
# Clone the repository
git clone https://github.com/Yara-95/Salmonella-amr-poultry-residue.git
cd Salmonella-amr-poultry-residue

# Install dependencies
pip install scikit-learn shap pandas matplotlib numpy

# Run the analysis
python project_AMR.py
________________________________________
Citation
If you use this dataset or code, please cite:
Al-Hallak Y. Predicting Human Salmonella Antimicrobial Resistance from 
Poultry Antibiotic Residues: A Simulation-Based Machine Learning 
Proof-of-Concept with One Health Integration. BMC Veterinary Research. 
Submitted 2026.
________________________________________
Important Note
This is a simulation-based proof-of-concept study. The dataset is entirely synthetic, and the results demonstrate pipeline feasibility, not a validated biological finding. Prospective validation on real, linked poultry-residue and human-clinical data is required before any translation to regulatory or screening practice.
________________________________________
Author
Yara Al-Hallak - Master’s Student in Bioinformatics, Syrian Virtual University - Senior Regulatory Affairs Specialist - Email: yara_377869@svuonline.org - GitHub: @Yara-95
________________________________________
License
This project is released under the MIT License. No restrictions apply to reuse.
________________________________________
Acknowledgements
The author gratefully acknowledges the Faculty of Veterinary Medicine at Hama University for the field observations and biological insights that informed the design of the simulation.
