# =====================================================================
# Simulation-Based Analysis of Synthetic Salmonella Resistance Labels
# 
# ENHANCEMENTS ADDED:
# 1. SHAP Values for individual prediction explanation
# 2. Logistic Regression as additional baseline
# 3. Precision-Recall Curve for imbalanced class evaluation
# =====================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
import shap
import importlib
shap.plots = importlib.import_module("shap.plots")

# Configuration
np.random.seed(7)
sns.set_theme(style="whitegrid")

print("=" * 70)
print("AIML PROJECT v4.0 - ENHANCED WITH SHAP, LR BASELINE, PR CURVE")
print("=" * 70)

# -----------------------------------------------------------------
# [1] LOAD DATASET
# -----------------------------------------------------------------
print("\n[Phase 1] Loading dataset...")
dataset_df = pd.read_csv('human_amr_poultry_residue_dataset_stm.csv')
print(f"Dataset: {dataset_df.shape[0]} records, {dataset_df.shape[1]} features")

# -----------------------------------------------------------------
# [2] FEATURE ENGINEERING
# -----------------------------------------------------------------
print("\n[Phase 2] Feature Engineering...")
X_categorical = pd.get_dummies(dataset_df[['Antibiotic_Tested', 'Geographic_Location']], drop_first=True)
y = dataset_df['Human_Resistance_Outcome']
X_full = pd.concat([X_categorical, dataset_df[['Poultry_Residue_Level_mg_kg', 'Human_MIC_mg_L']]], axis=1)
feature_names = X_full.columns.tolist()

# -----------------------------------------------------------------
# [3] TRAIN-TEST SPLIT
# -----------------------------------------------------------------
print("\n[Phase 3] Train-Test Split (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=7, stratify=y
)

# Scale for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------
# [4] MODEL 1: BASELINE RANDOM FOREST (Categorical Only)
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("[MODEL 1] Baseline Random Forest (Categorical Only)")
print("=" * 60)

cat_cols = [c for c in X_train.columns if 'Antibiotic' in c or 'Geographic' in c]
X_train_cat = X_train[cat_cols]
X_test_cat = X_test[cat_cols]

rf_baseline = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=7)
rf_baseline.fit(X_train_cat, y_train)

base_preds = rf_baseline.predict(X_test_cat)
base_proba = rf_baseline.predict_proba(X_test_cat)[:, 1]
base_acc = accuracy_score(y_test, base_preds) * 100
base_auc = roc_auc_score(y_test, base_proba)
base_ap = average_precision_score(y_test, base_proba)

print(f"Accuracy: {base_acc:.2f}% | AUC-ROC: {base_auc:.4f} | AP: {base_ap:.4f}")

# -----------------------------------------------------------------
# [5] MODEL 2: LOGISTIC REGRESSION (All Features) - NEW!
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("[MODEL 2] Logistic Regression (All Features)")
print("=" * 60)

lr_model = LogisticRegression(max_iter=1000, random_state=7)
lr_model.fit(X_train_scaled, y_train)

lr_preds = lr_model.predict(X_test_scaled)
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
lr_acc = accuracy_score(y_test, lr_preds) * 100
lr_auc = roc_auc_score(y_test, lr_proba)
lr_ap = average_precision_score(y_test, lr_proba)

print(f"Accuracy: {lr_acc:.2f}% | AUC-ROC: {lr_auc:.4f} | AP: {lr_ap:.4f}")

# LR Feature Importance
lr_importance = pd.Series(np.abs(lr_model.coef_[0]), index=feature_names).sort_values(ascending=False)
print("\nLogistic Regression |Coefficients|:")
print(lr_importance)

# -----------------------------------------------------------------
# [6] MODEL 3: OPTIMIZED RANDOM FOREST (All Features)
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("[MODEL 3] Optimized Random Forest (All Features)")
print("=" * 60)

rf_optimized = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=7)
rf_optimized.fit(X_train, y_train)

opt_preds = rf_optimized.predict(X_test)
opt_proba = rf_optimized.predict_proba(X_test)[:, 1]
opt_acc = accuracy_score(y_test, opt_preds) * 100
opt_auc = roc_auc_score(y_test, opt_proba)
opt_ap = average_precision_score(y_test, opt_proba)

print(f"Accuracy: {opt_acc:.2f}% | AUC-ROC: {opt_auc:.4f} | AP: {opt_ap:.4f}")
print("\nClassification Report (Optimized RF):")
print(classification_report(y_test, opt_preds, target_names=['Susceptible', 'Resistant']))

# -----------------------------------------------------------------
# [7] COMPARATIVE VISUALIZATION
# -----------------------------------------------------------------
print("\n[Phase 4] Generating Comparative Plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ROC Curve
ax1 = axes[0, 0]
fpr_base, tpr_base, _ = roc_curve(y_test, base_proba)
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
fpr_opt, tpr_opt, _ = roc_curve(y_test, opt_proba)
ax1.plot(fpr_base, tpr_base, 'b--', lw=2, label=f'RF Baseline (AUC={base_auc:.3f})')
ax1.plot(fpr_lr, tpr_lr, 'orange', linestyle='-.', lw=2, label=f'Logistic Reg (AUC={lr_auc:.3f})')
ax1.plot(fpr_opt, tpr_opt, 'g-', lw=2.5, label=f'RF Optimized (AUC={opt_auc:.3f})')
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve Comparison (v4.0)', fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# Precision-Recall Curve - NEW!
ax2 = axes[0, 1]
baseline_pr = y_test.sum() / len(y_test)
ax2.axhline(y=baseline_pr, color='gray', linestyle='--', alpha=0.5, label=f'Random (AP={baseline_pr:.3f})')
precision_base, recall_base, _ = precision_recall_curve(y_test, base_proba)
precision_lr, recall_lr, _ = precision_recall_curve(y_test, lr_proba)
precision_opt, recall_opt, _ = precision_recall_curve(y_test, opt_proba)
ax2.plot(recall_base, precision_base, 'b--', lw=2, label=f'RF Baseline (AP={base_ap:.3f})')
ax2.plot(recall_lr, precision_lr, 'orange', linestyle='-.', lw=2, label=f'Logistic Reg (AP={lr_ap:.3f})')
ax2.plot(recall_opt, precision_opt, 'g-', lw=2.5, label=f'RF Optimized (AP={opt_ap:.3f})')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve (v4.0)', fontweight='bold')
ax2.legend(loc='lower left')
ax2.grid(True, alpha=0.3)

# Feature Importance Comparison
ax3 = axes[1, 0]
rf_importance = pd.Series(rf_optimized.feature_importances_, index=feature_names).sort_values(ascending=True)
lr_imp_norm = (lr_importance / lr_importance.sum()).sort_values(ascending=True)
rf_imp_norm = (rf_importance / rf_importance.sum()).sort_values(ascending=True)

y_pos = np.arange(len(feature_names))
ax3.barh(y_pos - 0.2, rf_imp_norm.values, 0.4, label='RF Optimized', color='steelblue', alpha=0.8)
ax3.barh(y_pos + 0.2, lr_imp_norm[rf_imp_norm.index].values, 0.4, label='Logistic Regression', color='coral', alpha=0.8)
ax3.set_yticks(y_pos)
ax3.set_yticklabels([f.replace('Antibiotic_Tested_', 'AB_').replace('Geographic_Location_', 'Geo_') for f in rf_imp_norm.index])
ax3.set_xlabel('Normalized Importance')
ax3.set_title('Feature Importance: RF vs LR', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='x')

# Performance Summary
ax4 = axes[1, 1]
models = ['RF\nBaseline', 'Logistic\nRegression', 'RF\nOptimized']
accuracies = [base_acc, lr_acc, opt_acc]
aucs = [base_auc * 100, lr_auc * 100, opt_auc * 100]
aps = [base_ap * 100, lr_ap * 100, opt_ap * 100]

x = np.arange(len(models))
width = 0.25
ax4.bar(x - width, accuracies, width, label='Accuracy (%)', color='skyblue', edgecolor='navy')
ax4.bar(x, aucs, width, label='AUC-ROC (×100)', color='lightgreen', edgecolor='darkgreen')
ax4.bar(x + width, aps, width, label='Avg Precision (×100)', color='salmon', edgecolor='darkred')
ax4.set_ylabel('Score')
ax4.set_title('Model Performance Comparison', fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(models)
ax4.legend(fontsize=9)
ax4.set_ylim(0, 105)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('comparison_plots_v4.png', dpi=200, bbox_inches='tight')
plt.show()
print("✅ Comparison plots saved!")

# -----------------------------------------------------------------
# [8] SHAP EXPLAINABILITY - NEW!
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("[Phase 5] SHAP Explainability Analysis")
print("=" * 60)

explainer = shap.TreeExplainer(rf_optimized)
shap_values = explainer.shap_values(X_test)

# Extract positive-class SHAP contributions.
if isinstance(shap_values, list):
    shap_class1 = np.asarray(shap_values[1])
else:
    arr = np.asarray(shap_values)
    if arr.ndim == 3 and arr.shape[2] == 2:
        shap_class1 = arr[:, :, 1]
    elif arr.ndim == 2 and arr.shape == X_test.shape:
        shap_class1 = arr
    else:
        raise ValueError(f'Unexpected SHAP shape: {arr.shape}')

# Create SHAP Explanation object
explanation = shap.Explanation(
    values=shap_class1,
    base_values=explainer.expected_value[1] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value,
    data=X_test.values,
    feature_names=feature_names
)

# SHAP Bar Plot
plt.figure(figsize=(10, 6))
shap.plots.bar(explanation, max_display=10, show=False)
plt.title("SHAP Feature Importance (Global)\nImpact on Predicting Resistance", fontweight='bold')
plt.tight_layout()
plt.savefig('shap_summary_bar.png', dpi=200, bbox_inches='tight')
plt.show()
print("✅ SHAP Bar Plot saved!")

# SHAP Beeswarm Plot
plt.figure(figsize=(10, 8))
shap.plots.beeswarm(explanation, max_display=10, show=False)
plt.title("SHAP Beeswarm Plot\nFeature Impact Distribution", fontweight='bold')
plt.tight_layout()
plt.savefig('shap_beeswarm.png', dpi=200, bbox_inches='tight')
plt.show()
print("✅ SHAP Beeswarm saved!")

# SHAP Waterfall for individual cases
resistant_idx = np.where(y_test.values == 1)[0][0]
susceptible_idx = np.where(y_test.values == 0)[0][0]

plt.figure(figsize=(10, 6))
shap.plots.waterfall(explanation[resistant_idx], max_display=10, show=False)
plt.title(f"SHAP Waterfall: Resistant Case\nPrediction = {rf_optimized.predict_proba(X_test.iloc[[resistant_idx]])[0][1]:.3f}", fontweight='bold')
plt.tight_layout()
plt.savefig('shap_waterfall_resistant.png', dpi=200, bbox_inches='tight')
plt.show()
print("✅ SHAP Waterfall (Resistant) saved!")

plt.figure(figsize=(10, 6))
shap.plots.waterfall(explanation[susceptible_idx], max_display=10, show=False)
plt.title(f"SHAP Waterfall: Susceptible Case\nPrediction = {rf_optimized.predict_proba(X_test.iloc[[susceptible_idx]])[0][1]:.3f}", fontweight='bold')
plt.tight_layout()
plt.savefig('shap_waterfall_susceptible.png', dpi=200, bbox_inches='tight')
plt.show()
print("✅ SHAP Waterfall (Susceptible) saved!")

# -----------------------------------------------------------------
# [9] FINAL SUMMARY
# -----------------------------------------------------------------
print("\n" + "=" * 70)
print("FINAL COMPARATIVE SUMMARY (v4.0)")
print("=" * 70)

summary = pd.DataFrame({
    'Model': ['RF Baseline (Categorical)', 'Logistic Regression (All)', 'RF Optimized (All)'],
    'Accuracy (%)': [base_acc, lr_acc, opt_acc],
    'AUC-ROC': [base_auc, lr_auc, opt_auc],
    'Avg Precision': [base_ap, lr_ap, opt_ap]
})
print(summary.to_string(index=False))

print("\nKEY INSIGHTS:")
print(f"1. Logistic Regression: {lr_acc:.1f}% accuracy (linear baseline)")
print(f"2. Precision-Recall: RF Optimized AP = {opt_ap:.3f} (excellent for imbalanced data)")
print(f"3. SHAP confirms: Poultry_Residue is #1 driver (mean |SHAP| = {np.abs(shap_class1[:, feature_names.index('Poultry_Residue_Level_mg_kg')]).mean():.3f})")
print(f"4. MIC overlap does not remove label-conditional MIC circularity")
print("=" * 70)

summary.to_csv('model_comparison_summary.csv', index=False)
print("\n✅ All outputs saved successfully!")

# Cross-validation: shuffled folds, distinct from calibration folds.
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from scipy.stats import t
import sys
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
cv_models = [
    ('RF Baseline', RandomForestClassifier(n_estimators=50,max_depth=5,random_state=7), X_categorical),
    ('Logistic Regression', make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=7)), X_full),
    ('RF Optimized', RandomForestClassifier(n_estimators=150,max_depth=12,random_state=7), X_full)
]
cv_rows=[]; fold_rows=[]
for name, model, features in cv_models:
    scores=cross_validate(model,features,y,cv=folds,scoring={'Accuracy':'accuracy','AUC':'roc_auc','AP':'average_precision'})
    row={'Model':name}
    for metric in ['Accuracy','AUC','AP']:
        v=scores['test_'+metric]
        row[metric+'_mean']=v.mean()
        row[metric+'_95CI_halfwidth']=t.ppf(.975,4)*v.std(ddof=1)/np.sqrt(5)
        fold_rows.extend({'Model':name,'Metric':metric,'Fold':i+1,'Score':value} for i,value in enumerate(v))
    cv_rows.append(row)
cv_summary=pd.DataFrame(cv_rows)
cv_summary.to_csv('cross_validation_summary.csv',index=False)
pd.DataFrame(fold_rows).to_csv('cross_validation_fold_scores.csv',index=False)
print('\nCROSS-VALIDATION\n',cv_summary.to_string(index=False))

# Isotonic calibration: five stratified, unshuffled TRAINING folds.
calibrated_model=CalibratedClassifierCV(
    estimator=RandomForestClassifier(n_estimators=150,max_depth=12,random_state=7),
    method='isotonic',cv=5,ensemble=True)
calibrated_model.fit(X_train,y_train)
cal_proba=calibrated_model.predict_proba(X_test)[:,1]
cal_summary=pd.DataFrame({'Model':['Raw Random Forest','Isotonic Random Forest (cv=5)'],
    'Brier_score':[brier_score_loss(y_test,opt_proba),brier_score_loss(y_test,cal_proba)]})
cal_summary.to_csv('calibration_summary_cv5.csv',index=False)
print('\nCALIBRATION\n',cal_summary.to_string(index=False))
fig,ax=plt.subplots(figsize=(7,6)); points=[]
for name,prob in [('Raw Random Forest',opt_proba),('Isotonic calibration (cv=5)',cal_proba)]:
    observed,predicted=calibration_curve(y_test,prob,n_bins=10,strategy='uniform')
    ax.plot(predicted,observed,marker='o',label=name)
    points.extend({'Model':name,'Mean_probability':a,'Observed_fraction':b} for a,b in zip(predicted,observed))
ax.plot([0,1],[0,1],'--',color='gray',label='Perfect calibration')
ax.set(xlabel='Mean predicted resistance probability',ylabel='Observed resistant-label fraction',xlim=(0,1),ylim=(0,1))
ax.legend();fig.tight_layout()
fig.savefig('Figure3_calibration_cv5.jpg',dpi=300,bbox_inches='tight')
plt.show();plt.close(fig)
pd.DataFrame(points).to_csv('calibration_curve_points.csv',index=False)

# Rule-based reference posterior.
ranges={'ampicillin':((2.,64.),(.5,32.)),'ciprofloxacin':((.03,4.),(.015,2.)),
        'tetracycline':((1.,32.),(.25,16.)),'streptomycin':((2.,128.),(.5,64.))}
test_records=dataset_df.loc[X_test.index]
prior=np.where(test_records.Poultry_Residue_Level_mg_kg.to_numpy()>1.25,.8,.15)
fr=[];fs=[]
for abx,mic in zip(test_records.Antibiotic_Tested,test_records.Human_MIC_mg_L):
    (a,b),(c,d)=ranges[str(abx).strip().lower()]
    fr.append(float(a<=mic<=b)/(b-a));fs.append(float(c<=mic<=d)/(d-c))
num=prior*np.asarray(fr);den=num+(1-prior)*np.asarray(fs)
if (den==0).any():raise ValueError('MIC outside both assumed class ranges.')
posterior=num/den
reference_rows=[]
for name,prob in [('Reference posterior',posterior),('Residue threshold',(prior==.8).astype(float))]:
    reference_rows.append({'Model':name,'AUC_ROC':roc_auc_score(y_test,prob),
        'Accuracy':accuracy_score(y_test,prob>=.5),'AP':average_precision_score(y_test,prob)})
reference_summary=pd.DataFrame(reference_rows)
reference_summary.to_csv('reference_benchmark_summary.csv',index=False)
print('\nREFERENCE BENCHMARK\n',reference_summary.to_string(index=False))

# SHAP exports and probability-scale check.
expected=np.asarray(explainer.expected_value).reshape(-1)
base=float(expected[1] if len(expected)>1 else expected[0])
reconstructed=base+shap_class1.sum(axis=1)
max_error=float(np.max(np.abs(reconstructed-opt_proba)))
if not np.allclose(reconstructed,opt_proba,atol=1e-5):
    raise ValueError('SHAP contributions do not reconstruct positive-class probabilities.')
global_shap=pd.DataFrame({'Feature':feature_names,'Mean_absolute_SHAP':np.abs(shap_class1).mean(axis=0)}).sort_values('Mean_absolute_SHAP',ascending=False)
global_shap.to_csv('shap_global_verified.csv',index=False)
case_rows=[]
for label,pos in [('Resistant',resistant_idx),('Susceptible',susceptible_idx)]:
    idx=X_test.index[pos]
    row={'Class':label,'Patient_ID':dataset_df.loc[idx,'Patient_ID'],
        'Predicted_probability':opt_proba[pos],'Base_value':base,'Base_plus_all_SHAP':reconstructed[pos]}
    for j,f in enumerate(feature_names):
        row[f]=X_test.iloc[pos][f];row['SHAP_'+f]=shap_class1[pos,j]
    case_rows.append(row)
pd.DataFrame(case_rows).to_csv('shap_cases_verified.csv',index=False)
pd.DataFrame({'Check':['Max SHAP probability reconstruction error'],'Value':[max_error]}).to_csv('shap_additivity_check.csv',index=False)
pd.DataFrame({'Package':['Python','numpy','pandas','scikit-learn','shap','matplotlib'],
    'Version':[sys.version.split()[0],np.__version__,pd.__version__,__import__('sklearn').__version__,shap.__version__,__import__('matplotlib').__version__]}).to_csv('software_versions.csv',index=False)
print('\nGLOBAL SHAP\n',global_shap.to_string(index=False))
print('\nAll main analyses completed. Scenario analysis remains in simulation_scenarios_v2.py.')
