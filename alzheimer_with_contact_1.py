# -*- coding: utf-8 -*-
"""NeuroGene - Integrated Alzheimer Platform (HIERO 2026)."""
import base64
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

st.set_page_config(
    page_title="NeuroGene | Integrated System",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Schibsted+Grotesk:wght@400;500;600&display=swap');
html, body, .stApp, [class*="css"] { font-family:'Schibsted Grotesk', system-ui, sans-serif; color:#16202B; }
.stApp { background:#F2F5F7; }
h1, h2, h3, .serif { font-family:'Source Serif 4', Georgia, serif !important; font-weight:600; letter-spacing:-0.01em; }

/* إزالة تامة لأي شريط أبيض علوي وإخفاء الهيدر الافتراضي */
header[data-testid="stHeader"] { display: none !important; visibility: hidden !important; height: 0px !important; background: transparent !important; }
.stMainBlockContainer { padding-top: 0rem !important; }
#MainMenu, footer { visibility:hidden; }
.block-container { padding-top: 1rem !important; max-width:1240px; }
[data-testid="stSidebar"] { background:#E4EAEF; border-right:1px solid #C9D3DB; }

/* تنسيق وتوسيط النصوص في الصفحة الرئيسية */
.centered-header {
    text-align: center;
    margin-top: 10px;
    margin-bottom: 30px;
}
.centered-header h1 {
    color: #0D1520;
    font-size: 2.6rem;
    margin-bottom: 12px;
}
.centered-header p {
    color: #44525F;
    font-size: 1.15rem;
    max-width: 750px;
    margin: 0 auto;
    line-height: 1.6;
}

.mast { display:flex; justify-content:space-between; align-items:flex-end; gap:24px; flex-wrap:wrap; margin-bottom:1.4rem; }
.mast h1 { font-size:2rem; line-height:1.15; margin:0; padding:0; max-width:18ch; text-shadow: 0px 0px 1px;}
.mast p { margin:.6rem 0 0; max-width:58ch; color:#44525F; font-size:1.02rem; line-height:1.5; }
.mast .meta { color:#6B7A88; font-size:.9rem; text-align:right; }

.slide { background:#0D1520; border-radius:8px; padding:24px 26px 18px; }
.slide-grid { display:grid; grid-template-columns:repeat(auto-fill, 22px); gap:11px; justify-content:start; min-height:66px; }
.spot { width:16px; height:16px; border-radius:50%; background:var(--c); box-shadow:0 0 9px 1px var(--g); }
.spot.idle { background:#1B2837; box-shadow:none; }
.spot.dev { animation:develop .8s ease-out both; animation-delay:var(--d); }
@keyframes develop { from { opacity:0; transform:scale(.3); } to { opacity:1; transform:scale(1); } }
.slide-cap { display:flex; justify-content:space-between; align-items:center; gap:14px; flex-wrap:wrap; margin-top:18px; color:#8FA1B3; font-size:.86rem; }
.scale { display:flex; align-items:center; gap:10px; }
.scale i { display:block; width:170px; height:6px; border-radius:3px; background:linear-gradient(90deg,#1FAA8C,#465262,#D6336C); }

.verdict { font-size:1.55rem; line-height:1.35; margin:1.4rem 0 .4rem; max-width:34em; }
.verdict b { font-weight:600; }
.verdict .ad { color:#D6336C; } .verdict .ctl { color:#1FAA8C; } .verdict .unk { color:#6B7A88; }
.note { color:#5B6A78; font-size:.92rem; max-width:62ch; line-height:1.5; }
.stTabs [data-baseweb="tab-list"] { gap:28px; border-bottom:1px solid #C9D3DB; }
.stTabs [data-baseweb="tab"] { padding:10px 0; font-weight:500; }
.foot { color:#7C8A97; font-size:.82rem; margin-top:3rem; border-top:1px solid #D3DBE2; padding-top:12px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ Theme: logo colours, background, banner (all pages)
BG_OPACITY = 0.20   # background image opacity: 0.05 very faint ... 0.30 stronger

@st.cache_data
def b64(path_str):
    with open(path_str, "rb") as f:
        return base64.b64encode(f.read()).decode()

_bg_file, _logo_file = HERE / "bg.png", HERE / "NeuroGene.png"
_bg_css = ""
if _bg_file.exists():
    _bg_css = ('.stApp::before { content:""; position:fixed; inset:0; '
               'background:url("data:image/png;base64,' + b64(str(_bg_file)) + '") center/cover no-repeat; '
               'opacity:' + str(BG_OPACITY) + '; z-index:0; pointer-events:none; }')

THEME_CSS = """
<style>
:root { --teal-dark:#0B3C49; --teal:#1B7F8C; --blue:#2A5FA8; --gold:#C9A24B; }

/* background image */
.stApp { background:#f4f7fa; }
/*BG*/
.block-container, [data-testid="stMainBlockContainer"] { position:relative; z-index:1; }

/* centred logo */
.logo-wrap { display:flex; justify-content:center; margin-bottom:.5rem; }
.logo-wrap img { width:260px; max-width:60%; }

/* sidebar */
[data-testid="stSidebar"] { border-right:none; }
[data-testid="stSidebar"] > div:first-child { background:linear-gradient(180deg,#0B3C49 0%,#1B7F8C 55%,#2A5FA8 100%); }
[data-testid="stSidebar"] * { color:#ffffff !important; }
[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.3); }
[data-testid="stSidebar"] [data-baseweb="select"] > div { background:#d6eef0 !important; border:1px solid #9fd3d9; border-radius:10px; }
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-baseweb="select"] input { color:#0B3C49 !important; -webkit-text-fill-color:#0B3C49 !important; opacity:1 !important; font-weight:600; }
[data-testid="stSidebar"] [data-baseweb="select"] svg { fill:#0B3C49 !important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] section { background:rgba(255,255,255,.12); border:1px dashed rgba(255,255,255,.5); border-radius:10px; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] button { background:#ffffff; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] button * { color:#0B3C49 !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] { border:1px solid rgba(255,255,255,.35); border-radius:10px; background:rgba(255,255,255,.08); }

/* top logo row: partner logos on both sides of the centre logo */
.top-logos { display:grid; grid-template-columns:1fr auto 1fr; align-items:center; gap:22px; margin-bottom:.9rem; }
.top-logos { padding:0 .5rem; }
.top-logos .side { display:flex; align-items:center; gap:26px; flex-wrap:wrap; }
.top-logos .side.left { justify-content:flex-start; }
.top-logos .side.right { justify-content:flex-end; }
.top-logos .side img { width:auto; max-width:170px; object-fit:contain; }
.top-logos .center img { width:260px; max-width:100%; }
@media (max-width:900px) {
  .top-logos { grid-template-columns:1fr; gap:12px; }
  .top-logos .center { order:-1; display:flex; justify-content:center; }
  .top-logos .side.left, .top-logos .side.right { justify-content:center; }
}

/* circular sidebar logo + caption */
.side-logo { display:flex; justify-content:center; margin:.2rem 0 .7rem; }
.side-logo .circle { width:130px; height:130px; border-radius:50%; background:#ffffff; overflow:hidden;
                     border:3px solid #C9A24B; box-shadow:0 4px 14px rgba(0,0,0,.28); }
.side-logo img { width:100%; height:100%; object-fit:contain; }
.side-caption { text-align:center; font-weight:600; font-size:.95rem; line-height:1.35; margin:0 0 .4rem; }
.side-nav-hint { font-size:.92rem; line-height:1.4; margin:.2rem 0 .6rem; opacity:.95; }
.contact-sidebar { margin-top:1rem; padding-top:.9rem; border-top:1px solid rgba(255,255,255,.3); }
.contact-sidebar h4 { margin:0 0 .35rem; font-size:1.05rem; color:#fff; }
.contact-sidebar p { margin:0; line-height:1.5; color:#fff; }
.contact-sidebar a { color:#F6D889 !important; text-decoration:underline; font-weight:600; }
.contact-card { height:100%; padding:22px 18px; background:#fff; border:1px solid #dce3e9; border-radius:14px; box-shadow:0 5px 18px rgba(22,32,43,.06); text-align:center; }
.contact-card h3 { margin:.7rem 0 .25rem; font-size:1.2rem; color:#0B3C49; }
.contact-card .role { color:#536574; font-size:.91rem; line-height:1.5; min-height:5.2rem; }
.contact-card .email { display:block; margin-top:.8rem; overflow-wrap:anywhere; color:#1B7F8C; font-size:.9rem; }
.contact-avatar { width:112px; height:112px; margin:0 auto; border-radius:50%; display:flex; align-items:center; justify-content:center; background:linear-gradient(135deg,#0B3C49,#2A5FA8); color:white; font-size:1.7rem; font-weight:600; border:3px solid #C9A24B; }


/* top banner (every page) */
.hero, .mast {
    background:linear-gradient(120deg,#0B3C49 0%,#1B7F8C 60%,#2A5FA8 100%);
    border-bottom:4px solid var(--gold);
    border-radius:14px;
    padding:28px 28px;
    margin-bottom:1.2rem;
}
.hero { text-align:center; }
.hero h1, .mast h1 { color:#ffffff !important; margin:0; text-shadow:none; }
.hero h1 { font-size:2.2rem; }
.hero p, .mast p { color:#d6eef0 !important; margin:6px auto 0; }
.mast .meta { color:#cfe6e9 !important; }
.mast-center { position:relative; justify-content:center; text-align:center; }
.mast-center > div:first-child { width:100%; padding:0 150px; }
.mast-center h1 { max-width:none; margin:0 auto; }
.mast-center .meta { position:absolute; right:28px; top:50%; transform:translateY(-50%); text-align:right; }
@media (max-width:900px) {
  .mast-center > div:first-child { padding:0; }
  .mast-center .meta { position:static; transform:none; text-align:center; margin-top:10px; width:100%; }
}
</style>
""".replace("/*BG*/", _bg_css)
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ---- Top logo row: edit these lists to reorder / move logos between the two sides
TOP_LEFT_LOGOS = ["eg", "mans", "knowture"]     # file names without extension (png/webp/jpg/jpeg)
TOP_RIGHT_LOGOS = ["daad", "gr"]
TOP_LOGO_HEIGHT = {"eg": 44, "gr": 44, "mans": 64, "daad": 56, "knowture": 64}   # base height in px
TOP_LOGO_SCALE = 1.4                                                             # 1.0 = previous size; raise to enlarge all


def _find_logo(stem):
    for ext in ("png", "webp", "jpg", "jpeg"):
        f = HERE / f"{stem}.{ext}"
        if f.exists():
            return f
    return None


def _logo_img(path, height=None):
    ext = path.suffix.lower().lstrip(".")
    mime = "jpeg" if ext == "jpg" else ext
    style = f' style="height:{height}px"' if height else ""
    return f'<img src="data:image/{mime};base64,{b64(str(path))}"{style} alt="{path.stem}">'


def _logo_side(names):
    return "".join(_logo_img(f, round(TOP_LOGO_HEIGHT.get(n, 60) * TOP_LOGO_SCALE)) for n in names if (f := _find_logo(n)))


_center_logo = _logo_img(_logo_file) if _logo_file.exists() else ""
st.markdown(f'<div class="top-logos"><div class="side left">{_logo_side(TOP_LEFT_LOGOS)}</div>'
            f'<div class="center">{_center_logo}</div>'
            f'<div class="side right">{_logo_side(TOP_RIGHT_LOGOS)}</div></div>',
            unsafe_allow_html=True)

# ------------------------------------------------------------------ Helpers
@st.cache_resource
def load_gene_artifacts():
    model = joblib.load(HERE / "alzheimer_model.pkl")
    genes = list(joblib.load(HERE / "model_genes.pkl"))
    return getattr(model, "best_estimator_", model), genes

@st.cache_resource
def load_clinical_artifacts():
    model = joblib.load(HERE / "clinical_alzheimer.pkl")
    features = joblib.load(HERE / "clinical_features.pkl")
    return model, features

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

# ------------------------------------------------------------------ Sidebar Navigation & Centered Logo
SIDE_LOGO_UP = 10        # px: move the logo up inside the circle (raise to move it higher)
SIDE_LOGO_SCALE = 0.85   # 1.0 = full size; lower = smaller so the logo text is not clipped by the circle
SIDEBAR_LOGO = next((f for f in (HERE / "NeuroGene.png", HERE / "NeuroGene.jpeg", HERE / "NeuroGene.jpg") if f.exists()), None)

with st.sidebar:
    if SIDEBAR_LOGO is not None:
        _mime = "png" if SIDEBAR_LOGO.suffix.lower() == ".png" else "jpeg"
        st.markdown(f'<div class="side-logo"><div class="circle"><img src="data:image/{_mime};base64,{b64(str(SIDEBAR_LOGO))}" '
                    f'style="transform:translateY(-{SIDE_LOGO_UP}px) scale({SIDE_LOGO_SCALE})"></div></div>',
                    unsafe_allow_html=True)
    st.markdown('<p class="side-caption">NeuroGene: A Multimodal Framework for Alzheimer’s Disease Diagnosis</p>',
                unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p class="side-nav-hint">Please navigate through the different modules of our system from here</p>',
                unsafe_allow_html=True)
    page_options = ["Home Overview", "Gene Expression Classifier", "MRI Brain Scan Classifier", "Clinical Assessment", "Contact"]
    requested_page = st.query_params.get("page", "home")
    page_index = 4 if requested_page == "contact" else 0
    app_page = st.selectbox("Select Module:", page_options, index=page_index)
    st.markdown("---")
    st.markdown("""
    <div class="contact-sidebar">
      <h4>Contact</h4>
      <p>For inquiries, you can mail us <a href="?page=contact">here</a>.</p>
    </div>
    """, unsafe_allow_html=True)

# ================================================================== PAGE 0: HOME OVERVIEW
if "Home Overview" in app_page:
    st.markdown("""
    <div class="hero">
      <h1>NeuroGene: A Multimodal Framework for Alzheimer’s Disease Diagnosis</h1>
      <p>Advanced multi-modal intelligence platform for Alzheimer's disease diagnosis developed for HIERO 2026.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    st.subheader("Project Overview & Architecture")
    st.markdown("""
    **NeuroGene** is a state-of-the-art computational platform designed to support early and precise detection of Alzheimer's Disease. 
    Our research leverages a robust tri-modal intelligence architecture combining:
    1. **Gene Expression Profiling (`gene`):** Analyzes hippocampal microarray and gene expression data to detect molecular signatures and pathways associated with neurodegeneration.
    2. **MRI Brain Scan Classification (`mri`):** Employs advanced dimensionality reduction (PCA) and support vector machines (SVM) to classify structural brain MRI scans across clinical dementia stages.
    3. **Clinical Assessment Module (`clinical`):** Evaluates patient-specific clinical parameters, cognitive scores (MMSE), and diagnostic metrics using optimized Random Forest estimators.
    """)

# ================================================================== MODULE 1: GENE EXPRESSION
elif "Gene Expression" in app_page:
    try:
        est, GENES = load_gene_artifacts()
    except Exception as e:
        st.error(f"Missing file: {e}")
        st.stop()

    st.markdown("""
    <div class="mast mast-center">
      <div>
        <h1>Alzheimer's gene-expression classifier</h1>
        <p>Upload hippocampal expression profiles. Each sample gets a score and the genes that pushed it toward Alzheimer or control.</p>
      </div>
      <div class="meta">NeuroGene<br>HIERO 2026</div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Input Settings")
        upload = st.file_uploader("Expression CSV", type=["csv"], help="Rows are samples, columns are gene symbols.")
        band = st.slider("Inconclusive band around 0.5", 0.0, 0.25, 0.10, 0.01, help="Samples with p(AD) inside this band are not called.")
        run = st.button("Score samples", type="primary", use_container_width=True)
        with st.expander("File format"):
            st.markdown(f"- One row per sample, one column per gene symbol.\n- At least 2 samples.\n- At least {core.MIN_COVERAGE:.0%} of the {len(GENES)} model genes present.")

    if run and upload is not None:
        try:
            raw = pd.read_csv(upload)
            meta, Z, n_missing = core.prepare(raw, GENES)
            prob, contrib, sel, b0 = core.score(est, Z, GENES)
            names = (meta["ID_REF"].astype(str).tolist() if "ID_REF" in meta else [f"S{i + 1}" for i in range(len(Z))])
            st.session_state["res"] = dict(file=upload.name, meta=meta, prob=prob, contrib=contrib, sel=sel, b0=b0, names=names, n_missing=n_missing)
        except Exception as err:
            st.session_state.pop("res", None)
            st.error(f"Could not score this file. {err}")

    res = st.session_state.get("res")

    if res is None:
        st.markdown(slide_html(), unsafe_allow_html=True)
        st.markdown('<p class="note" style="margin-top:1.2rem">Start in the sidebar: choose a CSV, then press <b>Score samples</b>.</p>', unsafe_allow_html=True)
    else:
        prob, names, meta = res["prob"], res["names"], res["meta"]
        call = core.calls(prob, band)
        n_ad, n_ct, n_un = (int((call == c).sum()) for c in ("Alzheimer", "Control", "Inconclusive"))

        st.markdown(slide_html(prob, names), unsafe_allow_html=True)
        st.markdown(f'<p class="verdict serif"><b class="ad">{n_ad}</b> of {len(prob)} samples read as Alzheimer, '
                    f'<b class="ctl">{n_ct}</b> as control, <b class="unk">{n_un}</b> inconclusive.</p>'
                    f'<p class="note">{html.escape(res["file"])}'
                    + (f" &middot; {res['n_missing']} model genes were absent and set to cohort mean." if res["n_missing"] else "") + "</p>", unsafe_allow_html=True)

        tab_calls, tab_why, tab_cohort, tab_method = st.tabs(["Calls", "Why this call", "Cohort view", "Method and limits"])

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
                         column_config={"p(AD)": st.column_config.ProgressColumn("p(AD)", min_value=0.0, max_value=1.0, format="%.3f")})
            st.download_button("📥 Download CSV", table.to_csv(index=False).encode(), "neurogene_calls.csv", "text/csv")

        with tab_why:
            pick = st.selectbox("Sample", names, index=int(np.argmax(prob)))
            i = names.index(pick)
            c = res["contrib"][i]
            logit = c.sum() + res["b0"]
            st.markdown(f'<p class="note">Log-odds of Alzheimer for <b>{html.escape(pick)}</b> are <b>{logit:+.2f}</b> (p = {prob[i]:.2f}).</p>', unsafe_allow_html=True)
            o = np.argsort(c)
            pick_idx = np.concatenate([o[:10], o[-10:]])
            fig = go.Figure(go.Bar(x=c[pick_idx], y=res["sel"][pick_idx], orientation="h",
                                   marker_color=[MAG if v > 0 else TEAL for v in c[pick_idx]]))
            fig.update_xaxes(title="Contribution to log-odds", zeroline=True, zerolinecolor=INK)
            st.plotly_chart(plot_base(fig, 520), use_container_width=True)

        with tab_cohort:
            k = 30
            top = np.argsort(np.abs(res["contrib"]).mean(0))[-k:][::-1]
            ordr = np.argsort(prob)
            M = res["contrib"][ordr][:, top].T
            lim = float(np.abs(M).max()) or 1.0
            fig = go.Figure(go.Heatmap(
                z=M, x=[names[j] for j in ordr], y=res["sel"][top], zmin=-lim, zmax=lim,
                colorscale=[[0, TEAL], [0.5, "#F2F5F7"], [1, MAG]], colorbar=dict(title="log-odds", thickness=12)))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(plot_base(fig, 640), use_container_width=True)

        with tab_method:
            st.markdown("""
            **Model:** Standardized expression profiling, ANOVA feature selection, and L2-regularized logistic regression.  
            **Limits:** Research prototype for HIERO 2026. Not a diagnostic device.
            """)

# ================================================================== MODULE 2: MRI ANALYSIS
elif "MRI Brain Scan" in app_page:
    st.markdown("""
    <div class="mast mast-center">
      <div>
        <h1>MRI Brain Scan Classifier</h1>
        <p>Upload brain MRI scans to evaluate dementia stages using machine learning classification.</p>
      </div>
      <div class="meta">NeuroGene<br>HIERO 2026</div>
    </div>
    """, unsafe_allow_html=True)

    @st.cache_resource
    def load_mri_artifacts():
        mri_model = joblib.load(HERE / "svm_alzheimer.pkl")
        class_names = joblib.load(HERE / "svm_class_names.pkl")
        return mri_model, class_names

    try:
        mri_model, class_names = load_mri_artifacts()
    except Exception as e:
        mri_model, class_names = None, None

    # Short description for each class the SVM can output
    MRI_INFO = {
        "nondemented": ("Non-Demented", "#1FAA8C",
            "No signs of dementia on this scan. Brain structure looks typical, with no marked "
            "shrinkage beyond normal ageing."),
        "verymilddemented": ("Very Mild Demented", "#C9A24B",
            "Earliest stage. Occasional memory lapses (names, recent events) while daily life stays "
            "independent. MRI may show only slight shrinkage in memory regions such as the hippocampus."),
        "milddemented": ("Mild Demented", "#E08A3C",
            "Memory and thinking problems become noticeable: repeated questions, trouble planning, "
            "handling money or finding words. Some support is needed in daily life. Shrinkage of the "
            "hippocampus and temporal lobes is more visible."),
        "moderatedemented": ("Moderate Demented", "#D6336C",
            "Substantial memory loss and confusion, with help needed for daily activities such as "
            "dressing or cooking. Brain shrinkage is widespread and the ventricles are enlarged."),
    }

    def mri_key(name):
        return str(name).replace(" ", "").replace("_", "").lower()

    mri_file = st.file_uploader("Upload MRI Brain Scan Image", type=["jpg", "jpeg", "png"])

    if mri_file is not None:
        img = Image.open(mri_file).convert("L")

        # image and button in the centre of the page
        _, mid, _ = st.columns([3, 2, 3])
        with mid:
            st.image(img, caption="Uploaded MRI Scan", use_container_width=True)
            run_mri = st.button("🔍 Run MRI Classification", type="primary", use_container_width=True)

        if run_mri:
            if mri_model is not None:
                try:
                    img_resized = img.resize((128, 128))
                    # Normalisation: divide by 255.0 to match the training range
                    arr = np.array(img_resized, dtype=np.float32).flatten().reshape(1, -1) / 255.0
                    pred = mri_model.predict(arr)[0]
                    label_name = class_names[pred] if class_names and pred < len(class_names) else f"Class {pred}"

                    title, color, desc = MRI_INFO.get(
                        mri_key(label_name), (str(label_name), "#6B7A88", "No description available for this class."))
                    _, mid2, _ = st.columns([1, 3, 1])
                    with mid2:
                        st.markdown(
                            f'<div style="text-align:center;margin-top:1rem">'
                            f'<div style="font-size:.9rem;color:#6B7A88">Diagnostic prediction</div>'
                            f'<div class="serif" style="font-size:2rem;font-weight:600;color:{color}">{title}</div>'
                            f'<div style="margin:.8rem auto 0;max-width:560px;padding:14px 18px;text-align:left;'
                            f'background:rgba(255,255,255,.85);border-left:5px solid {color};border-radius:8px;'
                            f'color:#16202B;line-height:1.55">{desc}</div>'
                            f'<p class="note" style="margin:.8rem auto 0">Research prototype, not a medical diagnosis.</p>'
                            f'</div>', unsafe_allow_html=True)
                except Exception as ex:
                    st.error(f"Error during MRI prediction processing: {ex}")
            else:
                st.warning("⚠️ MRI model artifact (`svm_alzheimer.pkl`) not found in repository.")

    with st.expander("About the four classes"):
        for key in ("nondemented", "verymilddemented", "milddemented", "moderatedemented"):
            t, c, d = MRI_INFO[key]
            st.markdown(f'<p style="margin:.2rem 0"><b style="color:{c}">{t}</b>: {d}</p>', unsafe_allow_html=True)

# ================================================================== CONTACT PAGE
elif "Contact" in app_page:
    st.markdown("""
    <div class="mast mast-center">
      <div>
        <h1>Contact Our Team</h1>
        <p>For inquiries and collaboration, please contact our research team.</p>
      </div>
      <div class="meta">NeuroGene<br>HIERO 2026</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<p class="note">Meet the researchers behind NeuroGene. Email addresses below are clickable.</p>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        if (HERE / "photo_1.jpg").exists():
            st.image(str(HERE / "photo_1.jpg"), width=150)
        else:
            st.markdown('<div class="contact-avatar">AS</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="contact-card">'
             '<h3>Ass. Prof. Sara El-Metwally</h3>'
                        '<div class="role">Associate Professor of Computer Science, Faculty of Computers and Information, Mansoura University</div>'
                        '<a class="email" href="mailto:sarah_almetwally4@mans.edu.eg">sarah_almetwally4@mans.edu.eg</a>'
            
            '</div>', unsafe_allow_html=True
        )
    with c2:
       if (HERE / "photo_1.jpg").exists():
                   st.image(str(HERE / "photo_2.jpg"), width=150)
       else:
                   st.markdown('<div class="contact-avatar">AS</div>', unsafe_allow_html=True)
       st.markdown(
            '<div class="contact-card">'
            '<h3>Eng. Amany Sherif</h3>'
                        '<div class="role">Teaching Assistant, Computer Science Department, Faculty of Computers and Information, Mansoura University</div>'
                        '<a class="email" href="mailto:amanysherif@mans.edu.eg">amanysherif@mans.edu.eg</a>'
            '</div>', unsafe_allow_html=True
        )

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.markdown('<div class="contact-avatar">EE</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="contact-card">'
            '<h3>Eng. Eman Elsaeed</h3>'
            '<div class="role">Assistant Lecturer, Information Technology Department, Faculty of Computers and Information, Mansoura University</div>'
            '<a class="email" href="mailto:eman_elsaeed@mans.edu.eg">eman_elsaeed@mans.edu.eg</a>'
            '</div>', unsafe_allow_html=True
        )
    with c4:
        st.markdown('<div class="contact-avatar">MB</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="contact-card">'
            '<h3>Eng. Mohammed El-Barbeer</h3>'
            '<div class="role">Assistant Lecturer, Information Technology Department, Faculty of Computers and Information, Mansoura University</div>'
            '<span class="email">Email address not provided</span>'
            '</div>', unsafe_allow_html=True
        )

    st.markdown('<p style="margin-top:1.5rem"><a href="?page=home">← Back to Home Overview</a></p>', unsafe_allow_html=True)

# ================================================================== MODULE 3: CLINICAL ASSESSMENT
else:
    st.markdown("""
    <div class="mast mast-center">
      <div>
        <h1>Clinical Assessment Predictor</h1>
        <p>Input patient clinical metrics and cognitive test scores for Random Forest-based Alzheimer's diagnosis.</p>
      </div>
      <div class="meta">NeuroGene<br>HIERO 2026</div>
    </div>
    """, unsafe_allow_html=True)

    @st.cache_resource
    def get_clinical_model():
        model = joblib.load(HERE / "clinical_alzheimer.pkl")
        features = joblib.load(HERE / "clinical_features.pkl")
        return model, features

    try:
        clinical_model, clinical_features = get_clinical_model()
    except Exception as e:
        clinical_model, clinical_features = None, ['Age', 'M/F', 'Educ', 'SES', 'MMSE', 'CDR']

    with st.form("clinical_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 50, 95, 70, help="Patient age in years")
            gender = st.selectbox("Gender (M/F)", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
            educ = st.slider("Education (Educ - Years)", 0, 25, 12)
        with col2:
            ses = st.slider("Socioeconomic Status (SES)", 1, 5, 3)
            mmse = st.slider("Mini-Mental State Examination (MMSE)", 0, 30, 24, help="Cognitive score (0-30)")
            cdr = st.selectbox("Clinical Dementia Rating (CDR)", options=[0.0, 0.5, 1.0, 2.0], format_func=lambda x: str(x))

        submitted = st.form_submit_button("🔍 Run Clinical Prediction", type="primary", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame([[age, gender, educ, ses, mmse, cdr]], columns=clinical_features)
        
        if clinical_model is not None:
            try:
                pred_val = clinical_model.predict(input_data)[0]
                pred_proba = clinical_model.predict_proba(input_data)[0] if hasattr(clinical_model, "predict_proba") else None
                
                st.markdown("---")
                st.subheader("Diagnostic Results")
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    if pred_val == 1:
                        st.markdown('<p class="verdict serif"><b class="ad">Positive for Alzheimer / Dementia Risk</b></p>', unsafe_allow_html=True)
                    else:
                        st.markdown('<p class="verdict serif"><b class="ctl">Normal / Non-Demented Control</b></p>', unsafe_allow_html=True)
                    
                    if pred_proba is not None:
                        st.metric("Estimated Risk Probability", f"{pred_proba[1]:.3f}")
                
                with res_col2:
                    if pred_proba is not None:
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = pred_proba[1] * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Dementia Risk (%)"},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#D6336C" if pred_val == 1 else "#1FAA8C"},
                                'steps': [
                                    {'range': [0, 50], 'color': "#E4EAEF"},
                                    {'range': [50, 100], 'color': "#FAD2E1"}
                                ]
                            }
                        ))
                        fig_gauge.update_layout(height=230, margin=dict(l=20, r=20, t=30, b=10))
                        st.plotly_chart(fig_gauge, use_container_width=True)
                        
            except Exception as err:
                st.error(f"Error during clinical model prediction: {err}")
        else:
            st.warning("⚠️ Clinical model artifact (`clinical_alzheimer.pkl`) missing.")

st.markdown('<div class="foot">NeuroGene &copy; 2026 &middot; HIERO 2026 research prototype</div>', unsafe_allow_html=True)