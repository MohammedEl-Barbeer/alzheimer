"""Core helpers for the NeuroGene Streamlit app (no UI code here)."""
import numpy as np
import pandas as pd

META_COLS = ["ID_REF", "Patient", "Stage"]
MIN_COVERAGE = 0.95

TEAL, MID, MAG = (31, 170, 140), (70, 82, 98), (214, 51, 108)


def spot_rgb(p):
    """p(AD) -> colour. Teal = control, dim slate = no call, magenta = Alzheimer."""
    p = float(np.clip(p, 0, 1))
    a, b, t = (TEAL, MID, p / 0.5) if p < 0.5 else (MID, MAG, (p - 0.5) / 0.5)
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def prepare(df, genes):
    """Raw upload -> (meta, z-scored matrix aligned to model genes, n_missing_genes)."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    meta_cols = [c for c in META_COLS if c in df.columns]
    meta = df[meta_cols].reset_index(drop=True)

    X = df.drop(columns=meta_cols)
    X.columns = pd.Index(X.columns).astype(str).str.strip().str.upper()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.T.groupby(level=0).mean().T            # collapse duplicated gene symbols

    if len(X) < 2:
        raise ValueError("Upload at least 2 samples. The model was trained on per-gene "
                         "z-scored data, which cannot be computed from a single sample.")

    present = set(X.columns)
    missing = [g for g in genes if g not in present]
    coverage = 1 - len(missing) / len(genes)
    if coverage < MIN_COVERAGE:
        raise ValueError(f"Only {coverage:.0%} of the {len(genes)} model genes were found "
                         f"(need {MIN_COVERAGE:.0%}). Check that columns are gene symbols.")

    sd = X.std(ddof=1).replace(0, 1).fillna(1)   # same per-gene z-score as training
    Z = ((X - X.mean()) / sd).fillna(0)
    Z = Z.reindex(columns=genes, fill_value=0.0)
    return meta, Z, len(missing)


def score(est, Z, genes):
    """Return p(AD), per-gene contributions to the log-odds, selected gene names, intercept."""
    Xv = Z.values
    prob = est.predict_proba(Xv)[:, 1]
    sc, fs, clf = est.named_steps["sc"], est.named_steps["fs"], est.named_steps["clf"]
    mask = fs.get_support()
    Xs = sc.transform(Xv)[:, mask]
    contrib = Xs * clf.coef_[0]
    return prob, contrib, np.array(genes)[mask], float(clf.intercept_[0])


def calls(prob, band):
    lo, hi = 0.5 - band, 0.5 + band
    return np.where(prob > hi, "Alzheimer", np.where(prob < lo, "Control", "Inconclusive"))
