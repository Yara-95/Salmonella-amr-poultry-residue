"""
Multi-scenario robustness analysis (v2) -- replicated design.
Corrections relative to v1:
  * 'original' scenario now uses the exact MIC ranges (v1 widened them by 10% and clipped lower bounds to 0.001).
  * Class imbalance is produced by sampling WITHOUT replacement from a large pool (v1 sampled with replacement,
    which duplicated residue values across train/test: 49% of test residues also appeared in train).
  * MIC noise = probability m that an MIC is drawn from the OPPOSITE class range (interpretable overlap manipulation).
  * Each scenario is replicated R times (independent datasets) -> mean, SD, 2.5-97.5 percentile of test AUC.
  * Three models per replicate: A = all features, B = without MIC (isolates the residue signal), C = MIC only.
"""
import sys, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings; warnings.filterwarnings('ignore')

ABX = ['ampicillin', 'ciprofloxacin', 'tetracycline', 'streptomycin']
REG = ['A', 'B', 'C', 'D']
MIC = {'ampicillin': {'R': (2.0, 64.0), 'S': (0.5, 32.0)},
       'ciprofloxacin': {'R': (0.03, 4.0), 'S': (0.015, 2.0)},
       'tetracycline': {'R': (1.0, 32.0), 'S': (0.25, 16.0)},
       'streptomycin': {'R': (2.0, 128.0), 'S': (0.5, 64.0)}}
BASE_PREV = 0.65  # approx. prevalence of the base design

def p_resist(res, abx, tau, p_hi, p_lo, form, tau_by_abx, effect_abx):
    t = np.array([tau_by_abx[a] for a in abx]) if tau_by_abx else tau
    if form == 'step':
        p = np.where(res > t, p_hi, p_lo)
    elif form == 'logistic':
        p = p_lo + (p_hi - p_lo) / (1 + np.exp(-(res - t) / 0.5))
    elif form == 'linear':
        p = p_lo + (p_hi - p_lo) * (res - 0.02) / (5.0 - 0.02)
    if effect_abx is not None:  # residue effect only for one antibiotic; others constant prevalence
        p = np.where(np.array(abx) == effect_abx, p, BASE_PREV)
    return p

def draw_mic(rng, abx, y, mode, m_swap, true_null):
    out = np.empty(len(abx))
    for i, (a, k) in enumerate(zip(abx, y)):
        if true_null:
            lo, hi = MIC[a]['S'][0], MIC[a]['R'][1]
        else:
            cls = 'R' if k == 1 else 'S'
            if m_swap > 0 and rng.random() < m_swap: cls = 'S' if cls == 'R' else 'R'
            lo, hi = MIC[a][cls]
        out[i] = np.exp(rng.uniform(np.log(lo), np.log(hi))) if mode == 'loguniform' else rng.uniform(lo, hi)
    return out

def generate(rng, n=2500, tau=1.25, p_hi=0.80, p_lo=0.15, form='step', tau_by_abx=None, effect_abx=None,
             mic_mode='uniform', m_swap=0.0, true_null=False, ratio=None):
    pool = n if ratio is None else 12000
    res = rng.uniform(0.02, 5.0, pool); abx = rng.choice(ABX, pool); reg = rng.choice(REG, pool)
    y = (rng.random(pool) < p_resist(res, abx, tau, p_hi, p_lo, form, tau_by_abx, effect_abx)).astype(int)
    if ratio is not None:
        nr = int(n * ratio); ns = n - nr
        idx = np.concatenate([rng.choice(np.where(y == 1)[0], nr, replace=False),
                              rng.choice(np.where(y == 0)[0], ns, replace=False)]); rng.shuffle(idx)
        res, abx, reg, y = res[idx], abx[idx], reg[idx], y[idx]
    mic = draw_mic(rng, abx, y, mic_mode, m_swap, true_null)
    return pd.DataFrame({'R': res, 'M': mic, 'A': abx, 'G': reg, 'y': y})

def feats(df, cols):
    return pd.get_dummies(df[cols], drop_first=True)

def fit_auc(df, cols, seed):
    X = feats(df, cols); y = df.y
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=seed, stratify=y)
    rf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=seed, n_jobs=-1).fit(Xtr, ytr)
    p = rf.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p), accuracy_score(yte, p > .5)

SC = [
 ('S0 True null (no signal; MIC independent)', dict(p_hi=BASE_PREV, p_lo=BASE_PREV, true_null=True)),
 ('S1 Residue null, MIC label-conditional', dict(p_hi=BASE_PREV, p_lo=BASE_PREV)),
 ('S2 Weak step (0.55/0.45)', dict(p_hi=.55, p_lo=.45)),
 ('S3 Moderate step (0.65/0.35)', dict(p_hi=.65, p_lo=.35)),
 ('S4 Base design (0.80/0.15)', dict()),
 ('S5 Strong step (0.95/0.05)', dict(p_hi=.95, p_lo=.05)),
 ('S6 High MIC overlap (20% swapped)', dict(m_swap=0.20)),
 ('S7 Imbalance (80% resistant)', dict(ratio=0.80)),
 ('S8 Imbalance (20% resistant)', dict(ratio=0.20)),
 ('S9 Logistic dose-response', dict(form='logistic')),
 ('S10 Linear dose-response', dict(form='linear')),
 ('S11 Log-uniform MIC distribution', dict(mic_mode='loguniform')),
 ('S12 Residue effect only for ciprofloxacin', dict(effect_abx='ciprofloxacin')),
 ('S13 Antibiotic-specific thresholds', dict(tau_by_abx={'ampicillin': .6, 'ciprofloxacin': 1.25, 'tetracycline': 2.0, 'streptomycin': 3.0})),
]
if __name__ == '__main__':
    REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    rows = []
    for name, kw in SC:
        a, b, c, acc = [], [], [], []
        for r in range(REPS):
            rng = np.random.default_rng(1000 + r); df = generate(rng, **kw)
            A, ac = fit_auc(df, ['R', 'M', 'A', 'G'], r); B, _ = fit_auc(df, ['R', 'A', 'G'], r); C, _ = fit_auc(df, ['M', 'A', 'G'], r)
            a.append(A); b.append(B); c.append(C); acc.append(ac)
        f = lambda v: (np.mean(v), np.std(v, ddof=1), np.percentile(v, 2.5), np.percentile(v, 97.5))
        rows.append(dict(Scenario=name, Resistant_pct=round(100 * df.y.mean(), 1),
                         AUC_all=f(a)[0], SD_all=f(a)[1], lo_all=f(a)[2], hi_all=f(a)[3], Acc_all=np.mean(acc),
                         AUC_noMIC=f(b)[0], SD_noMIC=f(b)[1], lo_noMIC=f(b)[2], hi_noMIC=f(b)[3],
                         AUC_MIConly=f(c)[0], SD_MIConly=f(c)[1]))
        print(name, 'all=%.3f noMIC=%.3f MIConly=%.3f' % (np.mean(a), np.mean(b), np.mean(c)), flush=True)
        pd.DataFrame(rows).round(4).to_csv('scenario_results_v2.csv', index=False)
    print('DONE')
