# -*- coding: utf-8 -*-
"""NeuroGene - Integrated Alzheimer Platform (HIERO 2026)."""
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
header[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer { visibility:hidden; }
.block-container { padding-top:2.2rem; max-width:1240px; }
[data-testid="stSidebar"] { background:#E4EAEF; border-right:1px solid #C9D3DB; }
.logo-container { display: flex; justify-content: center; align-items: center; margin-bottom: 15px; }

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

.verdict { font-size:1.55rem; line-height:1.35; margin:1.4rem 0 .4rem; max-width:34em; }
.verdict b { font-weight:600; }
.verdict .ad { color:#D6336C; } .verdict .ctl { color:#1FAA8C; } .verdict .unk { color:#6B7A88; }
.note { color:#5B6A78; font-size:.92rem; max-width:62ch; line-height:1.5; }
.foot { color:#7C8A97; font-size:.82rem; margin-top:3rem; border-top:1px solid #D3DBE2; padding-top:12px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ Sidebar Navigation & Centered Logo
with st.sidebar:
    logo = HERE / "NeuroGene.png"
    if logo.exists():
        st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        st.image(Image.open(logo), width=130)
        st.markdown('</div>', unsafe_allow_html=True)
            
    st.subheader("Navigation | وحدات النظام")
    app_page = st.selectbox(
        "اختر واجهة العمل:",
        ["🧬 Gene Expression Classifier", "🧠 MRI Brain Scan Classifier"]
    )
    st.markdown("---")

# ================================================================== MODULE 1: GENE EXPRESSION
if "Gene Expression" in app_page:
    @st.cache_resource
    def load_gene_artifacts():
        model = joblib.load(HERE / "alzheimer_model.pkl")
        genes = list(joblib.load(HERE / "model_genes.pkl"))
        return getattr(model, "best_estimator_", model), genes

    try:
        est, GENES = load_gene_artifacts()
    except Exception as e:
        st.error(f"⚠️ خطأ في تحميل ملفات الجينات: {e}")
        st.stop()

    st.markdown("""
    <div class="mast">
      <div>
        <h1>Alzheimer's Gene-Expression Classifier</h1>
        <p>Upload hippocampal expression profiles. Each sample gets a score and the genes that pushed it toward Alzheimer or control.</p>
      </div>
      <div class="meta">NeuroGene<br>HIERO 2026</div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Gene Input Settings")
        upload = st.file_uploader("Expression CSV", type=["csv"], help="Rows are samples, columns are gene symbols.")
        band = st.slider("Inconclusive band around 0.5", 0.0, 0.25, 0.10, 0.01)
        run = st.button("Score samples", type="primary", use_container_width=True)

    if run and upload is not None:
        try:
            raw = pd.read_csv(upload)
            meta, Z, n_missing = core.prepare(raw, GENES)
            prob, contrib, sel, b0 = core.score(est, Z, GENES)
            names = (meta["ID_REF"].astype(str).tolist() if "ID_REF" in meta else [f"S{i + 1}" for i in range(len(Z))])
            st.session_state["res_gene"] = dict(file=upload.name, meta=meta, prob=prob, contrib=contrib, sel=sel, b0=b0, names=names, n_missing=n_missing)
        except Exception as err:
            st.session_state.pop("res_gene", None)
            st.error(f"Could not score this file. {err}")

    res = st.session_state.get("res_gene")
    if res is None:
        st.markdown(f'<div class="slide"><div class="slide-grid">{"".join("<i class=\"spot idle\"></i>" for _ in range(48))}</div><div class="slide-cap"><span>Upload a CSV in the sidebar and score it.</span></div></div>', unsafe_allow_html=True)
    else:
        prob, names, meta = res["prob"], res["names"], res["meta"]
        call = core.calls(prob, band)
        n_ad, n_ct, n_un = (int((call == c).sum()) for c in ("Alzheimer", "Control", "Inconclusive"))
        
        st.markdown(f'<p class="verdict serif"><b class="ad">{n_ad}</b> of {len(prob)} samples read as Alzheimer, <b class="ctl">{n_ct}</b> as control, <b class="unk">{n_un}</b> inconclusive.</p>', unsafe_allow_html=True)
        
        tab_calls, tab_why = st.tabs(["Calls", "Why this call"])
        with tab_calls:
            table = meta.copy()
            table.insert(0, "Sample", names)
            table["p(AD)"] = prob
            table["Call"] = call
            st.dataframe(table, hide_index=True, use_container_width=True)
            st.download_button("📥 Download CSV", table.to_csv(index=False).encode(), "neurogene_calls.csv", "text/csv")
        with tab_why:
            pick = st.selectbox("Sample", names, index=int(np.argmax(prob)))
            i = names.index(pick)
            c = res["contrib"][i]
            o = np.argsort(c)
            pick_idx = np.concatenate([o[:10], o[-10:]])
            fig = go.Figure(go.Bar(x=c[pick_idx], y=res["sel"][pick_idx], orientation="h", marker_color=[MAG if v > 0 else TEAL for v in c[pick_idx]]))
            st.plotly_chart(fig, use_container_width=True)

# ================================================================== MODULE 2: MRI ANALYSIS
else:
    st.markdown("""
    <div class="mast">
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

    mri_file = st.file_uploader("Upload MRI Brain Scan Image", type=["jpg", "jpeg", "png"])
    
    if mri_file is not None:
        img = Image.open(mri_file).convert("L")
        st.image(img, caption="Uploaded MRI Scan", width=300)
        
        if st.button("🔍 Run MRI Classification", type="primary"):
            if mri_model is not None:
                try:
                    # تجهيز الصورة ومعالجتها لتتوافق مع مدخلات نموذج PCA/SVM المرفق
                    img_resized = img.resize((64, 64))
                    arr = np.array(img_resized).flatten().reshape(1, -1)
                    pred = mri_model.predict(arr)[0]
                    # تحديد اسم الفئة إن كانت مخزنة كقائمة أو مصفوفة
                    label_name = class_names[pred] if class_names and pred < len(class_names) else f"Class {pred}"
                    
                    st.markdown(f"### النتيجة التشخيصية: **{label_name}**")
                except Exception as ex:
                    st.error(f"حدث خطأ أثناء معالجة الصورة بالنموذج: {ex}")
            else:
                st.warning("⚠️ نموذج الـ MRI (`svm_alzheimer.pkl`) غير متوفر في المستودع.")

st.markdown('<div class="foot">NeuroGene &copy; 2026 &middot; HIERO 2026 research prototype</div>', unsafe_allow_html=True)