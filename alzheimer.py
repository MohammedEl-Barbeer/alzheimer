# -*- coding: utf-8 -*-
"""NeuroGene - Alzheimer gene-expression classifier (HIERO 2026)."""
import html
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import ad_core as core

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
INK, PAPER, SLIDE = "#16202B", "#F2F5F7", "#0D1520"
TEAL, MAG = "#1FAA8C", "#D6336C"

st.set_page_config(page_title="NeuroGene | Alzheimer expression classifier",
                   page_icon="🧬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Schibsted+Grotesk:wght@400;500;600&display=swap');
html, body, .stApp, [class*="css"] { font-family:'Schibsted Grotesk', system-ui, sans-serif; color:#16202B; }
.stApp { background:#F2F5F7; }
h1, h2, h3, .serif { font-family:'Source Serif 4', Georgia, serif !important; font-weight:600; letter-spacing:-0.01em; }
header[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer { visibility:hidden; }
.block-container { padding-top:2.2rem; max-width:1240px; }
[data-testid="stSidebar"] { background:#E4EAEF; border-right:1px solid #C9D3DB; }

.mast { display:flex; justify-content:space-between; align-items:flex-end; gap:24px; flex-wrap:wrap; margin-bottom:1.4rem; }
.mast h1 { font-size:2.15rem; line-height:1.15; margin:0; padding:0; max-width:18ch; }
.mast p { margin:.6rem 0 0; max-width:58ch; color:#44525F; font-size:1.02rem; line-height:1.5; }
.mast .meta { color:#6B7A88; font-size:.9rem; text-align:right; }

.slide { background:#0D1520; border-radius:8px; padding:24px 26px 18px; }
.slide-grid { display:grid; grid-template-columns:repeat(auto-fill, 22px); gap:11px; justify-content:start; min-height:66px; }
.spot { width:16px; height:16px; border-radius:50%; background:var(--c); box-shadow:0 0 9px 1px var(--g); }
.spot.idle { background:#1B2837; box-shadow:none; }
.spot.dev { animation:develop .8s ease-out both; animation-delay:var(--d); }
@keyframes develop { from { opacity:0; transform:scale(.3); } to { opacity:1; transform:scale(1); } }
@media (prefers-reduced-motion: reduce) { .spot.dev { animation:none; } }
.slide-cap { display:flex; justify-content:space-between; align-items:center; gap:14px; flex-wrap:wrap; margin-top:18px; color:#8FA1B3; font-size:.86rem; }
.scale { display:flex; align-items:center; gap:10px; }
.scale i { display:block; width:170px; height:6px; border-radius:3px; background:linear-gradient(90deg,#1FAA8C,#465262,#D6336C); }

.verdict { font-size:1.55rem; line-height:1.35; margin:1.4rem 0 .4rem; max-width:34em; }
.verdict b { font-weight:600; }
.verdict .ad { color:#D6336C; } .verdict .ctl { color:#1FAA8C; } .verdict .unk { color:#6B7A88; }
.note { color:#5B6A78; font-size:.92rem; max-width:62ch; line-height:1.5; }
.stTabs [data-baseweb="tab-list"] { gap:28px; border-bottom:1px solid #C9D3DB; }
.stTabs [data-baseweb="tab"] { padding:10px 0; font-weight:500; }
.foot { color:#7C8A97; font-size:.82rem; margin-top:3rem; border-top:1px solid #D3DBE2; padding-top:12px; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
@st.cache_resource
def load_artifacts():
    model = joblib.load(HERE / "alzheimer_model.pkl")
    est = getattr(model, "best_estimator_", model)
    return est, int(est.named_steps["sc"].n_features_in_)


def prepare_positional(df, n_feat):
    """Use only alzheimer_model.pkl: gene columns are matched BY POSITION, not by name.

    The uploaded CSV must contain exactly n_feat gene columns, in the same order as at training time.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    meta_cols = [c for c in core.META_COLS if c in df.columns]
    meta = df[meta_cols].reset_index(drop=True)

    X = df.drop(columns=meta_cols).apply(pd.to_numeric, errors="coerce")
    if len(X) < 2:
        raise ValueError("Upload at least 2 samples. The model was trained on per-gene "
                         "z-scored data, which cannot be computed from a single sample.")
    if X.shape[1] != n_feat:
        raise ValueError(f"The file has {X.shape[1]} gene columns but the model needs exactly {n_feat}, "
                         "in the same order as training.")

    # make column names unique so they can be shown in the plots
    seen, names = {}, []
    for c in X.columns:
        seen[c] = seen.get(c, 0) + 1
        names.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
    X.columns = names

    sd = X.std(ddof=1).replace(0, 1).fillna(1)
    Z = ((X - X.mean()) / sd).fillna(0)
    return meta, Z, names


def plot_base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Schibsted Grotesk, sans-serif", color=INK, size=13))
    fig.update_xaxes(showline=True, linecolor="#9AA8B5", gridcolor="#DCE3E9")
    fig.update_yaxes(showline=True, linecolor="#9AA8B5", gridcolor="#DCE3E9")
    return fig


def slide_html(probs=None, names=None):
    if probs is None:
        spots = "".join('<i class="spot idle"></i>' for _ in range(48))
        cap = "Each spot will be one sample. Upload a file and score it to develop the slide."
    else:
        out = []
        for i, p in enumerate(probs):
            r, g, b = core.spot_rgb(p)
            tip = html.escape(f"{names[i]}: p(AD) = {p:.2f}")
            out.append(f'<i class="spot dev" title="{tip}" style="--c:rgb({r},{g},{b});'
                       f'--g:rgba({r},{g},{b},.55);--d:{min(i * 14, 1400)}ms"></i>')
        spots = "".join(out)
        cap = f"{len(probs)} samples. Hover a spot for its sample ID and score."
    return (f'<div class="slide"><div class="slide-grid">{spots}</div>'
            f'<div class="slide-cap"><span>{cap}</span>'
            f'<span class="scale">Control<i></i>Alzheimer &nbsp;(dim = no clear call)</span></div></div>')


# ------------------------------------------------------------------ masthead
st.markdown("""
<div class="mast">
  <div>
    <h1>Alzheimer's gene-expression classifier</h1>
    <p>Upload hippocampal expression profiles. Each sample gets a score and the genes that pushed it toward Alzheimer or control.</p>
  </div>
  <div class="meta">NeuroGene<br>HIERO 2026</div>
</div>
""", unsafe_allow_html=True)

try:
    est, N_FEAT = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Missing file: {Path(e.filename).name}. Keep alzheimer_model.pkl "
             "in the same folder as alzheimer.py.")
    st.stop()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    logo = HERE / "NeuroGene.jpeg"
    if logo.exists():
        st.image(Image.open(logo), width=150)
    st.subheader("Input")
    upload = st.file_uploader("Expression CSV", type=["csv"],
                              help="Rows are samples, columns are gene symbols.")
    band = st.slider("Inconclusive band around 0.5", 0.0, 0.25, 0.10, 0.01,
                     help="Samples with p(AD) inside this band are not called.")
    run = st.button("Score samples", type="primary", use_container_width=True)
    with st.expander("File format"):
        st.markdown(f"- One row per sample, one column per gene symbol.\n"
                    f"- Optional columns: `ID_REF`, `Patient` (yes/no), `Stage`.\n"
                    f"- At least 2 samples.\n"
                    f"- Exactly {N_FEAT} gene columns, in the same order as the training data "
                    f"(columns are matched by position, not by name).")

if run and upload is not None:
    try:
        raw = pd.read_csv(upload)
        meta, Z, gene_names = prepare_positional(raw, N_FEAT)
        n_missing = 0
        prob, contrib, sel, b0 = core.score(est, Z, gene_names)
        names = (meta["ID_REF"].astype(str).tolist() if "ID_REF" in meta
                 else [f"S{i + 1}" for i in range(len(Z))])
        st.session_state["res"] = dict(file=upload.name, meta=meta, prob=prob, contrib=contrib,
                                       sel=sel, b0=b0, names=names, n_missing=n_missing)
    except Exception as err:
        st.session_state.pop("res", None)
        st.error(f"Could not score this file. {err}")
elif run:
    st.warning("Choose a CSV file in the sidebar first.")

res = st.session_state.get("res")

# ------------------------------------------------------------------ empty state
if res is None:
    st.markdown(slide_html(), unsafe_allow_html=True)
    st.markdown('<p class="note" style="margin-top:1.2rem">Start in the sidebar: choose a CSV, then press '
                '<b>Score samples</b>. The method and its limits are described in the last tab after scoring.</p>',
                unsafe_allow_html=True)
    st.stop()

# ------------------------------------------------------------------ results
prob, names, meta = res["prob"], res["names"], res["meta"]
call = core.calls(prob, band)
n_ad, n_ct, n_un = (int((call == c).sum()) for c in ("Alzheimer", "Control", "Inconclusive"))

st.markdown(slide_html(prob, names), unsafe_allow_html=True)
st.markdown(f'<p class="verdict serif"><b class="ad">{n_ad}</b> of {len(prob)} samples read as Alzheimer, '
            f'<b class="ctl">{n_ct}</b> as control, <b class="unk">{n_un}</b> inconclusive.</p>'
            f'<p class="note">{html.escape(res["file"])}'
            + (f" &middot; {res['n_missing']} model genes were absent and set to the cohort mean."
               if res["n_missing"] else "") + "</p>", unsafe_allow_html=True)

tab_calls, tab_why, tab_cohort, tab_method = st.tabs(
    ["Calls", "Why this call", "Cohort view", "Method and limits"])

# ---- Calls
with tab_calls:
    order = np.argsort(prob)
    fig = go.Figure(go.Bar(
        x=[names[i] for i in order], y=prob[order],
        marker_color=[f"rgb{core.spot_rgb(p)}" for p in prob[order]],
        hovertemplate="%{x}<br>p(AD) = %{y:.3f}<extra></extra>"))
    fig.add_hrect(y0=0.5 - band, y1=0.5 + band, fillcolor="#9AA8B5", opacity=0.18, line_width=0)
    fig.add_hline(y=0.5, line_dash="dot", line_color="#6B7A88")
    fig.update_yaxes(title="p(Alzheimer)", range=[0, 1])
    fig.update_xaxes(showticklabels=len(prob) <= 40, title="Samples, ordered by score")
    st.plotly_chart(plot_base(fig, 320), use_container_width=True)

    table = meta.copy()
    table.insert(0, "Sample", names)
    table["p(AD)"] = prob
    table["Call"] = call
    st.dataframe(table, hide_index=True, use_container_width=True,
                 column_config={"p(AD)": st.column_config.ProgressColumn(
                     "p(AD)", min_value=0.0, max_value=1.0, format="%.3f")})
    st.download_button("Download calls as CSV", table.to_csv(index=False).encode(),
                       "neurogene_calls.csv", "text/csv")

# ---- Why this call
with tab_why:
    pick = st.selectbox("Sample", names, index=int(np.argmax(prob)))
    i = names.index(pick)
    c = res["contrib"][i]
    logit = c.sum() + res["b0"]
    st.markdown(f'<p class="note">Log-odds of Alzheimer for <b>{html.escape(pick)}</b> are '
                f'<b>{logit:+.2f}</b> (p = {prob[i]:.2f}). It is the sum of the bars below over all '
                f'{len(c)} genes the model uses, plus an intercept of {res["b0"]:+.2f}. '
                f'The 10 strongest genes in each direction are shown.</p>', unsafe_allow_html=True)
    o = np.argsort(c)
    pick_idx = np.concatenate([o[:10], o[-10:]])
    vals, genes_ = c[pick_idx], res["sel"][pick_idx]
    fig = go.Figure(go.Bar(x=vals, y=genes_, orientation="h",
                           marker_color=[MAG if v > 0 else TEAL for v in vals],
                           hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
    fig.update_xaxes(title="Contribution to log-odds (right = toward Alzheimer, left = toward control)",
                     zeroline=True, zerolinecolor=INK)
    st.plotly_chart(plot_base(fig, 520), use_container_width=True)

# ---- Cohort view
with tab_cohort:
    k = 30
    top = np.argsort(np.abs(res["contrib"]).mean(0))[-k:][::-1]
    ordr = np.argsort(prob)
    M = res["contrib"][ordr][:, top].T
    lim = float(np.abs(M).max()) or 1.0
    fig = go.Figure(go.Heatmap(
        z=M, x=[names[j] for j in ordr], y=res["sel"][top], zmin=-lim, zmax=lim,
        colorscale=[[0, TEAL], [0.5, "#F2F5F7"], [1, MAG]],
        colorbar=dict(title="log-odds", thickness=12),
        hovertemplate="%{y} in %{x}: %{z:+.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(showticklabels=len(prob) <= 40, title="Samples, control-like on the left")
    st.markdown(f'<p class="note">The {k} genes that move the scores most across this cohort. Magenta pushes '
                f'a sample toward Alzheimer, teal toward control. A block of one colour across neighbouring '
                f'samples is a shared expression pattern.</p>', unsafe_allow_html=True)
    st.plotly_chart(plot_base(fig, 640), use_container_width=True)

    if "Patient" in meta.columns:
        obs = (meta["Patient"].astype(str).str.lower().str.strip() == "yes").astype(int).values
        if len(set(obs)) == 2:
            st.markdown("**Agreement with the `Patient` column**")
            pred = (prob >= 0.5).astype(int)
            a, b = st.columns(2)
            a.metric("AUC", f"{roc_auc_score(obs, prob):.3f}")
            b.metric("Balanced accuracy", f"{balanced_accuracy_score(obs, pred):.3f}")
            st.dataframe(pd.crosstab(pd.Series(np.where(obs == 1, "Alzheimer", "Control"), name="Recorded"),
                                     pd.Series(np.where(pred == 1, "Alzheimer", "Control"), name="Predicted")),
                         use_container_width=True)
            st.markdown('<p class="note">If these samples were part of model training, these numbers are '
                        'optimistic and say nothing about new patients.</p>', unsafe_allow_html=True)

# ---- Method and limits
with tab_method:
    st.markdown(f"""
**Model.** Per-gene z-scoring across the uploaded samples, then standardisation, selection of the top
{len(res['sel'])} genes by ANOVA F-score, and L2-regularised logistic regression. The model reads
{N_FEAT} genes (the shared, higher-variance genes across the training cohorts), matched by column position.

**Explanations.** The classifier is linear, so every score is an exact sum of per-gene contributions.
The bars in *Why this call* are those terms, not an approximation.

**Evidence.** On external cohorts the pre-declared pipeline reached AUC 0.57 (trained on GSE5281, tested
on the local cohort) and 0.82 (trained on the local cohort, tested on GSE5281). Training sets were small
(tens of samples), so confidence intervals are wide.

**Limits.**
- Scores depend on the uploaded batch, because z-scoring uses the cohort's own mean and spread. A very
  small or unbalanced upload shifts every score.
- Platform and brain-region differences between cohorts reduced transfer performance in validation.
- This is a research prototype. It is not a diagnostic device and must not guide clinical decisions.
""")

st.markdown('<div class="foot">NeuroGene &copy; 2026 &middot; HIERO 2026 research prototype</div>',
            unsafe_allow_html=True)
