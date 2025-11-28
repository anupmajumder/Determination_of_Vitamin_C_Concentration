# Config
FILE_PATH    = "./EC_Coated_VitC_newData.xlsx"
WVN_COL      = "Wavenumber (cm-1)"
TARGET_VAR   = "VC"        # predict Vitamin C amount
N_INTERVALS  = 25         # iPLS intervals across spectrum
PLS_MAX_COMP = 10          # max components to try
INNER_CV     = 5           # inner CV for iPLS selection (KFold)
RANDOM_SEED  = 0


import numpy as np
import pandas as pd
raw_data  = pd.read_excel(FILE_PATH)
raw_data

import os, re, warnings
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score, make_scorer

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

assert WVN_COL in raw_data.columns, f"Missing '{WVN_COL}'. Available columns: {raw_data.columns.tolist()[:8]}"

# all spectral columns are "not wavenumber"
spec_cols = [c for c in raw_data.columns if c != WVN_COL]

# parse each spectral column into (pattern, VC, std)
parsed = []
for c in spec_cols:
    s = str(c)
    # example pattern: "sample 5 mg 0.2"
    m = re.match(r"(.+?)\s([\d.]+)\s*mg\s*([\d.]+)$", s)
    if m:
        pattern, vc, std = m.groups()
        pattern = f"{pattern} {vc} mg {std}"
        parsed.append((pattern, float(vc), float(std)))
    else:
        # fallback: take first two numbers
        nums = re.findall(r"([\d.]+)", s)
        if len(nums) >= 2:
            vc, std = float(nums[0]), float(nums[1])
            parsed.append((s, vc, std))
        else:
            raise ValueError(f"Could not parse VC/std from column header: '{s}'")

ys = pd.DataFrame(parsed, columns=["pattern", "VC", "std"])
# keep parsed entries correspond to actual Excel columns
ys = ys[ys["pattern"].isin(spec_cols)].reset_index(drop=True)

ys

# raw rows = wavenumbers, columns = spectra -> transpose it to samples x features
X   = raw_data[ys["pattern"].tolist()].T.to_numpy(dtype=float)   # (n_samples, n_wvn)
wvn = raw_data[WVN_COL].to_numpy(dtype=float)                    # (n_wvn,)
Y   = ys[[TARGET_VAR]].to_numpy(dtype=float)                # (n_samples, 1)

# group by VC concentration for LOGO-CV
groups = ys["VC"].to_numpy()
# Create a new column for group based on VC concentrations
ys['group'] = ys['VC'].astype('category').cat.codes + 1  # Convert VC to group codes starting from 1

# Print the dataframe showing pattern, VC, std, and group
print(ys[['pattern', 'VC', 'std', 'group']].to_string())  # Display group

# sanity checks
assert X.shape[0] == Y.shape[0], "Samples mismatch between X and Y."
assert X.shape[1] == wvn.shape[0], "Features in X != length of wavenumber axis."
# Group Count Validation
n_samples, n_feat = X.shape
n_groups = len(np.unique(groups))
assert n_groups >= 2, "Need at least 2 unique groups for GroupKFold."

print(f"Loaded: samples={n_samples}, features={n_feat}, groups={n_groups}")


# SNV (Standard Normal Variate)
def snv(Xm):
    Xm = np.asarray(Xm, dtype=float)
    mu = Xm.mean(axis=1, keepdims=True)
    sd = Xm.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (Xm - mu) / sd

# Savitzky–Golay Smoothing Filter
def apply_savgol(Xm):
    Xm = np.asarray(Xm, dtype=float)
    # choose a window length based on number of vars
    wl = 11 if Xm.shape[1] >= 11 else (Xm.shape[1] if Xm.shape[1] % 2 == 1 else max(1, Xm.shape[1]-1))
    if wl < 3:
        return Xm
    return savgol_filter(Xm, window_length=wl, polyorder=2, axis=1)

SNV     = FunctionTransformer(snv, validate=False)
SAVGOL  = FunctionTransformer(apply_savgol, validate=False)

def pooled_rmse(y_true, y_pred):
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


gkf = GroupKFold(n_splits=n_groups)
max_comp = int(max(2, min(PLS_MAX_COMP, X.shape[1]-1, X.shape[0]-1)))

best_rmse_full, best_r2_full, best_n_full = np.inf, -np.inf, None
cv_true_full, cv_pred_full = [], []

rmse_curve = []
for n_comp in range(2, max_comp+1):
    # Building the preprocessing pipeline
    pipe = Pipeline([
        ("snv", SNV),
        ("sg", SAVGOL),
        ("pls", PLSRegression(n_components=n_comp))
    ])
    fold_true, fold_pred = [], []
    for tr, te in gkf.split(X, Y, groups):
        pipe.fit(X[tr], Y[tr])
        yhat = pipe.predict(X[te])
        fold_true.append(Y[te]); fold_pred.append(yhat)
    yt = np.vstack(fold_true); yp = np.vstack(fold_pred)
    rmse = pooled_rmse(yt, yp)
    rmse_curve.append((n_comp, rmse))
    if rmse < best_rmse_full:
        best_rmse_full = rmse
        best_r2_full   = r2_score(yt, yp)
        best_n_full    = n_comp
        cv_true_full, cv_pred_full = yt, yp

print(f"[PLS full] best_n={best_n_full}, RMSE={best_rmse_full:.4f}, R²={best_r2_full:.4f}")

import numpy as np
np.linspace(0, 23, 6, dtype=int)


def split_intervals(n_vars, n_intervals):
    n_intervals = min(n_intervals, n_vars)
    edges = np.linspace(0, n_vars, n_intervals+1, dtype=int)
    return [(edges[i], edges[i+1]) for i in range(n_intervals)]

def pls_rmsecv_inner(Xm, ym, n_components=5, inner_cv=INNER_CV, seed=RANDOM_SEED):
    # inner CV is plain KFold (not grouped) just for feature selection
    inner_cv = min(inner_cv, max(2, Xm.shape[0]))
    kf = KFold(n_splits=inner_cv, shuffle=True, random_state=seed)
    pls = PLSRegression(n_components=n_components)
    neg_mse = cross_val_score(pls, Xm, ym,
                              scoring=make_scorer(mean_squared_error, greater_is_better=False),
                              cv=kf)
    return float(np.sqrt(-neg_mse.mean()))

def forward_ipls_mask(Xm, ym, n_intervals=50, n_components=5, wvn=None):
    n_samp, n_vars = Xm.shape
    intervals = split_intervals(n_vars, n_intervals)
    remaining = list(range(len(intervals)))
    mask = np.zeros(n_vars, dtype=bool)
    history = []
    last_rmse = np.inf
    idxes = []

    while remaining:
        candidates = []
        for idx in remaining:
            s, e = intervals[idx]
            trial_mask = mask.copy()
            trial_mask[s:e] = True
            X_sel = Xm[:, trial_mask]
            if X_sel.shape[1] < 2 or X_sel.shape[0] <= 2:
                continue
            n_comp = int(min(n_components, X_sel.shape[1], X_sel.shape[0]-1))
            if n_comp < 1:
                continue
            rmse = pls_rmsecv_inner(X_sel, ym, n_components=n_comp)
            candidates.append((rmse, idx, s, e))
        if not candidates:
            break
        candidates.sort(key=lambda t: t[0])
        rmse, idx, s, e = candidates[0]
        # stop if no improvement
        if rmse >= last_rmse - 1e-8:
            break
        # accept interval
        mask[s:e] = True
        remaining.remove(idx)
        history.append(rmse)
        idxes.append(idx)
        last_rmse = rmse

    # Get the selected wavenumbers based on the mask
    selected_wavenumbers = wvn[mask] if wvn is not None else None

    if selected_wavenumbers is not None:
        print(f"Selected wavenumbers: {selected_wavenumbers}")
    else:
        print("No intervals selected or no improvement found.")

    return mask, history, selected_wavenumbers, (s,e), intervals, idxes  # Return selected wavenumbers


# choose a modest components cap for selection
n_comp_sel = int(min(5, max_comp))

max_comp_sel = int(max(2, PLS_MAX_COMP))

best_rmse_ipls, best_r2_ipls, best_n_ipls = np.inf, -np.inf, None
cv_true_ipls, cv_pred_ipls = [], []

for n_comp in range(2, max_comp_sel+1):
    fold_true, fold_pred = [], []
    pipe_tr = Pipeline([
            ("snv", SNV),
            ("sg", SAVGOL),
            #("pls", PLSRegression(n_components=n_comp_use))
        ])
    X_pp = pipe_tr.fit_transform(X)
    for tr, te in gkf.split(X, Y, groups):
        ipls_mask, ipls_hist, selected_wavenumbers, (s,e), intervals, idxes = forward_ipls_mask(
            X_pp[tr],
            Y[tr],
            n_intervals=N_INTERVALS,
            n_components=n_comp_sel,
            wvn=wvn
        )
        if not ipls_mask.any():
            print("[iPLS] No interval improved RMSE; falling back to full spectrum for iPLS model.")
            ipls_mask = np.ones(X.shape[1], dtype=bool)
        
        X_sel = X[tr][:, ipls_mask]
        
        n_comp_use = int(min(n_comp, X_sel.shape[1], X_sel.shape[0]))
        
        pipe = Pipeline([
            ("snv", SNV),
            ("sg", SAVGOL),
            ("pls", PLSRegression(n_components=n_comp_use))
        ])
        
        pipe.fit(X_sel, Y[tr])
        yhat = pipe.predict(X[te][:, ipls_mask])
        fold_true.append(Y[te]); fold_pred.append(yhat)
    
    yt = np.vstack(fold_true); yp = np.vstack(fold_pred)
    rmse = pooled_rmse(yt, yp)
    if rmse < best_rmse_ipls:
        best_rmse_ipls = rmse
        best_r2_ipls   = r2_score(yt, yp)
        best_n_ipls    = n_comp  # Store the original n_comp value
        cv_true_ipls, cv_pred_ipls = yt, yp
        
print(f"[iPLS→PLS] best_n={best_n_ipls}, RMSE={best_rmse_ipls:.4f}, R²={best_r2_ipls:.4f}")

mean_spectrum = X.mean(axis=0)

plt.figure(figsize=(10, 6))
plt.plot(wvn, mean_spectrum, color='black', lw=2, label='Mean Spectrum')

for wn in selected_wavenumbers:
    plt.axvline(x=wn, color='blue', linestyle='--', alpha=0.7)

plt.xlabel("Wavenumber (cm$^{-1}$)", fontsize=12)
plt.ylabel("Absorbance", fontsize=12)
plt.title("Mean Spectrum with Selected Wavenumbers Highlighted", fontsize=14)

plt.legend(["Mean Spectrum", "Selected Wavenumbers"], loc='upper right')
plt.gca().invert_xaxis()

plt.tight_layout()
plt.show()

#use_ipls = best_rmse_ipls < best_rmse_full
use_ipls = True
label = "iPLS→PLS" if use_ipls else "PLS (full)"

yt = cv_true_ipls.ravel() if use_ipls else cv_true_full.ravel()
yp = cv_pred_ipls.ravel() if use_ipls else cv_pred_full.ravel()

rmse = best_rmse_ipls if use_ipls else best_rmse_full
r2   = best_r2_ipls   if use_ipls else best_r2_full
best_n = best_n_ipls if use_ipls else best_n_full

print("\n================ RESULTS ================")
print(f"PLS (full spectrum):   RMSE={best_rmse_full:.4f}, R²={best_r2_full:.4f}, best_n={best_n_full}")
print(f"iPLS→PLS (selected):   RMSE={best_rmse_ipls:.4f}, R²={best_r2_ipls:.4f}, best_n={best_n_ipls}")
print(f"Chosen for plot: {label} (components={best_n})")
print("========================================\n")

# 1) full-spectrum plot
yt_full = cv_true_full.ravel()
yp_full = cv_pred_full.ravel()

plt.figure(figsize=(5.6, 5.6))
plt.scatter(yt_full, yp_full, s=36, alpha=0.85, color='blue', label='Full Spectrum PLS')
lo, hi = min(yt_full.min(), yp_full.min()), max(yt_full.max(), yp_full.max())
plt.plot([lo, hi], [lo, hi], color='red', linewidth=2, label='y = x')

plt.xlabel("Actual VC (mg)", fontsize=12)
plt.ylabel("Predicted VC (mg)", fontsize=12)
plt.title(
    f"PLS (Full Spectrum) — Pred vs Actual VC\n"
    f"RMSE = {best_rmse_full:.3f}, R² = {best_r2_full:.3f}, Components = {best_n_full}",
    fontsize=11
)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 2) chosen model plot
plt.figure(figsize=(5.6, 5.6))
plt.scatter(yt, yp, s=36, alpha=0.85)
lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
plt.plot([lo, hi], [lo, hi], linewidth=2)
plt.xlabel("Actual VC (mg)")
plt.ylabel("Predicted VC (mg)")
plt.title(f"{label} — Pred vs Actual VC\nRMSE={rmse:.3f}, R²={r2:.3f}")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()




