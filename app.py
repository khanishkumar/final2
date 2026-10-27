"""AI-Based Heat Exchanger Fouling Prediction, Performance Monitoring & Maintenance Support Dashboard.
Prototype: automotive oil-cooler-based, PHE-like heat-exchanger (hot: water, cold: coolant)."""
from __future__ import annotations
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from hxlib import data as D, engineering as E, explain as X, maintenance as R, models as M

warnings.filterwarnings("ignore")
st.set_page_config(page_title="HX Fouling Dashboard", page_icon="🔧", layout="wide")
st.markdown("""<style>
.kpi{border:1px solid rgba(128,128,128,.35);border-radius:10px;padding:12px 14px;margin-bottom:8px;
background:rgba(128,128,128,.07)}
.kpi .kl{font-size:.78rem;opacity:.75}.kpi .kv{font-size:1.45rem;font-weight:700}
.kpi .ku{font-size:.8rem;font-weight:400;opacity:.75}.kpi .ks{font-size:.75rem;opacity:.7}
.kpi.good{border-left:6px solid #2e9e5b}.kpi.bad{border-left:6px solid #d64545}
.kpi.warn{border-left:6px solid #e0a030}.kpi.na{border-left:6px solid #888}
</style>""", unsafe_allow_html=True)

NONE = "— not available —"
SECTIONS = ["1 · Data Input", "2 · Data Preprocessing", "3 · Engineering Performance",
            "4 · Fouling Prediction & XAI", "5 · Maintenance Support"]
WF_A, WF_B, WF_C = ("A · Clean reference row from the dataset", "B · Separately supplied Uclean",
                    "C · Rf column already present in the dataset")
REF_ONE, REF_MED = "Single selected clean row", "Median U of all rows labelled Clean"
ROW_ALL = "All rows"
DEFAULTS = {"p_section": SECTIONS[0], "p_workflow": WF_A, "p_ref_mode": REF_ONE, "p_row_filter": ROW_ALL,
            "p_pos": 0, "p_ref_idx": 0, "p_uclean_sup": 1500.0, "p_uclean_unit": "W/(m²·K)",
            "p_confirmed": False, "p_thr_val": 0.043, "p_thr_unit": "m²·K/kW", "p_sev_med": 1.0,
            "p_sev_high": 3.0, "p_iqr_k": 1.5, "p_dedup": False, "p_missing": M.MISSING_OPTS[0],
            "p_outlier": M.OUTLIER_OPTS[0], "p_test_size": 0.25, "p_seed": 42, "p_trees": 200,
            "p_split": "Stratified random", "p_models": M.MODEL_NAMES, "p_shap_n": 500,
            "p_u_unit": "W/(m²·K)", "p_rf_unit": "m²·K/W"}
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
for _k in [k for k in st.session_state.keys() if k.startswith("p_")]:   # keep widget values across pages
    st.session_state[_k] = st.session_state[_k]


# ------------------------------------------------------------------ helpers
def fmt(x, spec=".4g", unit=""):
    try:
        if x is None or not np.isfinite(float(x)):
            return "n/a"
        return f"{float(x):{spec}}" + (f" {unit}" if unit else "")
    except (TypeError, ValueError):
        return "n/a"


def kpi(label, value, unit="", sub="", tone=""):
    st.markdown(f'<div class="kpi {tone}"><div class="kl">{label}</div><div class="kv">{value}'
                f'<span class="ku"> {unit}</span></div><div class="ks">{sub}</div></div>', unsafe_allow_html=True)


def mapping() -> dict:
    out = {}
    for r in D.ROLES:
        v = st.session_state.get(f"p_map_{r}")
        out[r] = None if v in (None, NONE) else v
    return out


def role_unit(role, col):
    return st.session_state.get(f"p_unit_{role}") or D.ROLES[role][1]


def feature_units(mp) -> dict:
    return {c: role_unit(r, c) for r, c in mp.items() if c and r not in ("target", "time", "u", "rf")} | \
           ({mp["u"]: st.session_state["p_u_unit"]} if mp.get("u") else {})


def ingest(up):
    sig = (up.name, up.size)
    if st.session_state.get("file_sig") == sig:
        return
    try:
        df = D.load_csv(up)
    except ValueError as e:
        st.error(f"Upload failed: {e}")
        return
    for k in [k for k in st.session_state.keys() if k.startswith("p_") and k != "p_section"]:
        del st.session_state[k]
    for k in ("ml", "shap_cache"):
        st.session_state.pop(k, None)
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)
    st.session_state.update(df=df, file_sig=sig, file_name=up.name)
    det = D.detect_columns(df)
    for r, c in det.items():
        st.session_state[f"p_map_{r}"] = c or NONE
        st.session_state[f"p_unit_{r}"] = D.guess_unit(c, r)
    st.session_state["p_u_unit"] = D.guess_u_unit(det["u"])
    st.session_state["p_rf_unit"] = D.guess_rf_unit(det["rf"])
    if det["target"]:
        vals = sorted(df[det["target"]].dropna().astype(str).unique().tolist())
        c, f = D.guess_labels(vals)
        st.session_state["p_clean_val"], st.session_state["p_fouled_val"] = c, f
        st.session_state["p_ref_mode"] = REF_MED
        first = df.index[df[det["target"]].astype(str) == str(c)]
        if len(first):
            st.session_state["p_ref_idx"] = int(first[0])
    st.session_state["p_workflow"] = WF_A if det["u"] else WF_B


def require_data():
    df = st.session_state.get("df")
    if df is None:
        st.info("Upload a CSV in **Section 1 · Data Input** to begin.")
        st.stop()
    return df


def require_mapping():
    df = require_data()
    mp = mapping()
    errs, _ = D.validate_mapping(df, mp)
    if errs or not st.session_state.get("p_confirmed"):
        st.warning("Please resolve mapping errors and tick **confirm column mappings** in Section 1 first.")
        for e in errs:
            st.error(e)
        st.stop()
    return df, mp


def label_values(df, mp):
    if not mp["target"]:
        return None, None
    opts = sorted(df[mp["target"]].dropna().astype(str).unique().tolist())
    c, f = st.session_state.get("p_clean_val"), st.session_state.get("p_fouled_val")
    if c not in opts or f not in opts:
        c, f = D.guess_labels(opts)
        st.session_state["p_clean_val"], st.session_state["p_fouled_val"] = c, f
    return c, f


def filtered_indices(df, mp):
    c, f = label_values(df, mp)
    flt = st.session_state["p_row_filter"]
    if flt == ROW_ALL or not mp["target"]:
        return df.index.to_numpy()
    val = c if flt.startswith("Only Clean") else f
    return df.index[df[mp["target"]].astype(str) == str(val)].to_numpy()


def selected_index(df, mp) -> int:
    idx = filtered_indices(df, mp)
    if len(idx) == 0:
        return int(df.index[0])
    pos = min(int(st.session_state["p_pos"]), len(idx) - 1)
    return int(idx[pos])


def engineering_state(df, mp, idx) -> dict:
    """Single source of truth for Rf, threshold, severity (used by Sections 3, 4, 5)."""
    wf, ss = st.session_state["p_workflow"], st.session_state
    thr = E.threshold_to_si(ss["p_thr_val"], ss["p_thr_unit"])
    ev = {"wf": wf, "thr_si": thr, "notes": [], "uclean_si": None, "uf_si": None, "ref_desc": "",
          "res": E.RfResult(False, message="Not evaluated.")}
    row = df.loc[idx]
    ufac = E.U_UNITS[ss["p_u_unit"]]
    if wf == WF_C:
        if not mp["rf"]:
            ev["res"] = E.RfResult(False, message="No Rf column is mapped.")
        else:
            ev["res"] = E.rf_from_column(row[mp["rf"]], ss["p_rf_unit"])
            ev["ref_desc"] = f"Rf read from column '{mp['rf']}' ({ss['p_rf_unit']})"
    else:
        uf = E.to_float(row[mp["u"]]) * ufac if (mp["u"] and E.to_float(row[mp["u"]]) is not None) else None
        uc = None
        if wf == WF_A:
            if not mp["u"]:
                ev["notes"].append("Map the U column in Section 1.")
            elif ss["p_ref_mode"] == REF_ONE:
                ridx = int(min(ss["p_ref_idx"], len(df) - 1))
                v = E.to_float(df.loc[ridx, mp["u"]])
                uc = v * ufac if v is not None else None
                ev["ref_desc"] = f"Clean reference = row {ridx}"
                c, f = label_values(df, mp)
                if mp["target"] and str(df.loc[ridx, mp["target"]]) != str(c):
                    ev["notes"].append(f"Reference row {ridx} is labelled '{df.loc[ridx, mp['target']]}', not Clean ('{c}').")
                if uc is None:
                    ev["notes"].append(f"Reference row {ridx} has no valid U value.")
                else:
                    ev["notes"].append("A single-row reference is noisy; consider the median-of-clean option.")
            else:
                c, _ = label_values(df, mp)
                if not mp["target"]:
                    ev["notes"].append("Median-of-clean reference needs a mapped label column.")
                elif mp["u"]:
                    s = df.loc[df[mp["target"]].astype(str) == str(c), mp["u"]].dropna()
                    s = s[s > 0]
                    if len(s):
                        uc = float(s.median()) * ufac
                        ev["ref_desc"] = f"Clean reference = median U of {len(s)} rows labelled Clean ('{c}')"
                    else:
                        ev["notes"].append("No valid U values among Clean-labelled rows.")
        else:  # WF_B
            uc = float(ss["p_uclean_sup"]) * E.U_UNITS[ss["p_uclean_unit"]]
            ev["ref_desc"] = "Clean reference = user-supplied Uclean"
        ev["uf_si"], ev["uclean_si"] = uf, uc
        ev["res"] = E.compute_rf(uf, uc)
    res = ev["res"]
    ev["cmp"] = E.compare_threshold(res.rf_si if res.available else None, thr)
    ev["sev"] = E.severity_from_rf(res.rf_si if res.available else None, thr,
                                   ss["p_sev_med"], ss["p_sev_high"])
    return ev


def label_text(df, mp, idx):
    if not mp["target"]:
        return None
    c, f = label_values(df, mp)
    v = str(df.loc[idx, mp["target"]])
    return "Clean" if v == str(c) else "Fouled" if v == str(f) else f"Other ({v})"


# ------------------------------------------------------------------ Section 1
def page_input():
    st.header("Section 1 · Data Input")
    up = st.file_uploader("Upload CSV dataset", type=["csv"])
    if up is not None:
        with st.spinner("Reading file…"):
            ingest(up)
    df = st.session_state.get("df")
    if df is None:
        st.info("Upload a CSV file to begin. Nothing is assumed about column names or label encoding.")
        return
    st.success(f"Loaded **{st.session_state['file_name']}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicate rows", int(df.duplicated().sum()))
    st.subheader("Dataset preview")
    st.dataframe(df.head(100), width="stretch")
    with st.expander("Column names and data types", expanded=False):
        st.dataframe(D.column_summary(df), width="stretch")

    st.subheader("Column mapping (auto-detected — please verify)")
    st.caption("Roles were guessed from column names only. Choose “not available” for anything the dataset lacks; "
               "no value will be invented.")
    cols = [NONE] + list(df.columns)
    grid = st.columns(3)
    for i, (role, (label, _)) in enumerate(D.ROLES.items()):
        with grid[i % 3]:
            st.selectbox(label, cols, key=f"p_map_{role}")
            if role == "u":
                st.selectbox("Unit of U", list(E.U_UNITS), key="p_u_unit")
            elif role == "rf":
                st.selectbox("Unit of Rf", list(E.RF_UNITS), key="p_rf_unit")
            elif role not in D.NON_NUMERIC_ROLES:
                st.text_input("Unit (display only)", key=f"p_unit_{role}")
    mp = mapping()
    errs, warns = D.validate_mapping(df, mp)
    for e in errs:
        st.error(e)
    for w in warns:
        st.warning(w)
    if mp["target"] and not errs:
        st.markdown("**Target label mapping** — do not assume 0 = Clean / 1 = Fouled without checking.")
        opts = sorted(df[mp["target"]].dropna().astype(str).unique().tolist())
        label_values(df, mp)
        vc = df[mp["target"]].astype(str).value_counts(dropna=False).rename("rows").to_frame()
        a, b, c = st.columns([1, 1, 1])
        a.dataframe(vc)
        b.selectbox("Value meaning **Clean**", opts, key="p_clean_val")
        c.selectbox("Value meaning **Fouled**", opts, key="p_fouled_val")
        if st.session_state["p_clean_val"] == st.session_state["p_fouled_val"]:
            st.error("Clean and Fouled must be different values.")
    st.checkbox("I confirm these column mappings, units and label meanings", key="p_confirmed",
                disabled=bool(errs))
    if errs or not st.session_state["p_confirmed"]:
        return

    st.subheader("Row selection")
    cfil = st.columns([1, 1, 2])
    opts = [ROW_ALL] + (["Only Clean-labelled rows", "Only Fouled-labelled rows"] if mp["target"] else [])
    if st.session_state["p_row_filter"] not in opts:
        st.session_state["p_row_filter"] = ROW_ALL
    cfil[0].selectbox("Restrict row list", opts, key="p_row_filter")
    idxs = filtered_indices(df, mp)
    st.session_state["p_pos"] = min(int(st.session_state["p_pos"]), max(len(idxs) - 1, 0))
    cfil[1].number_input(f"Position in list (0 – {max(len(idxs) - 1, 0)})", min_value=0,
                         max_value=max(len(idxs) - 1, 0), step=1, key="p_pos")
    idx = selected_index(df, mp)
    cfil[2].info(f"Selected dataset row index **{idx}** (CSV line {idx + 2} including the header).")

    st.subheader("Clean reference selection")
    wf_opts = [WF_A, WF_B] + ([WF_C] if mp["rf"] else [])
    if st.session_state["p_workflow"] not in wf_opts:
        st.session_state["p_workflow"] = wf_opts[0]
    st.radio("Rf workflow", wf_opts, key="p_workflow")
    wf = st.session_state["p_workflow"]
    if wf == WF_A:
        if not mp["u"]:
            st.error("Workflow A needs a mapped U column.")
        st.radio("Clean reference method", [REF_ONE, REF_MED], key="p_ref_mode", horizontal=True)
        if st.session_state["p_ref_mode"] == REF_ONE and mp["u"]:
            ca, cb = st.columns([2, 1])
            ca.number_input("Clean reference row index", min_value=0, max_value=len(df) - 1, step=1, key="p_ref_idx")
            if mp["target"]:
                c, _ = label_values(df, mp)
                first = df.index[df[mp["target"]].astype(str) == str(c)]
                if len(first):
                    cb.button("Use first Clean-labelled row", on_click=lambda f=int(first[0]): st.session_state.update(p_ref_idx=f))
            r = int(st.session_state["p_ref_idx"])
            st.caption(f"Reference row {r}: U = {fmt(df.loc[r, mp['u']], '.5g', st.session_state['p_u_unit'])}"
                       + (f", label = {df.loc[r, mp['target']]}" if mp["target"] else ""))
    elif wf == WF_B:
        ca, cb = st.columns(2)
        ca.number_input("Supplied Uclean (must come from a verified clean baseline)", min_value=0.0, step=10.0, key="p_uclean_sup")
        cb.selectbox("Unit of Uclean", list(E.U_UNITS), key="p_uclean_unit")
        st.warning("Make sure this Uclean uses the same heat-transfer-area basis as the dataset's U.")
    else:
        st.info("Rf is read directly from the mapped column; its unit was set above.")

    st.subheader("Selected-row details")
    row = df.loc[idx]
    rmap = {c: r for r, c in mp.items() if c}
    det = pd.DataFrame({"column": row.index, "value": [row[c] for c in row.index],
                        "role": [D.ROLES[rmap[c]][0] if c in rmap else "(unmapped)" for c in row.index],
                        "unit": [st.session_state["p_u_unit"] if rmap.get(c) == "u"
                                 else st.session_state["p_rf_unit"] if rmap.get(c) == "rf"
                                 else role_unit(rmap[c], c) if c in rmap and rmap[c] not in D.NON_NUMERIC_ROLES else ""
                                 for c in row.index]})
    st.dataframe(det.astype({"value": str}), width="stretch", hide_index=True)
    lab = label_text(df, mp, idx)
    if lab:
        st.info(f"Actual label in dataset (not a model prediction): **{lab}**")
    st.success("Rows and mappings are set. Continue to Sections 2–5 from the sidebar.")


# ------------------------------------------------------------------ Section 2
def default_features(df, mp):
    roles = [r for r in D.ROLES if r not in ("target", "time", "rf")]
    return [mp[r] for r in roles if mp[r]]


def page_preprocess():
    st.header("Section 2 · Data Preprocessing")
    df, mp = require_mapping()
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != mp["target"]]
    st.subheader("Data quality overview")
    miss = pd.DataFrame({"missing": df.isna().sum(), "missing %": (df.isna().mean() * 100).round(2)})
    a, b = st.columns([1, 1])
    a.dataframe(miss, width="stretch")
    fig = px.bar(miss.reset_index(), x="index", y="missing %", title="Missing values per column (%)")
    fig.update_layout(xaxis_title="", height=330)
    b.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    c1.metric("Duplicate rows", int(df.duplicated().sum()))
    c2.metric("Rows with ≥1 missing value", int(df.isna().any(axis=1).sum()))
    st.markdown("**Data types**")
    st.dataframe(D.column_summary(df)[["column", "dtype", "unique"]], width="stretch", hide_index=True)

    st.subheader("Outlier detection (IQR)")
    st.slider("IQR multiplier k", 1.0, 3.0, step=0.5, key="p_iqr_k")
    ot = M.iqr_table(df, numeric, st.session_state["p_iqr_k"])
    st.dataframe(ot, width="stretch", hide_index=True)
    colsel = st.selectbox("Inspect distribution of", numeric)
    st.plotly_chart(px.box(df, y=colsel, points=False, height=320), width="stretch")
    pins = D.boundary_pinned(df, numeric)
    with st.expander("Values pinned exactly at a column's minimum/maximum"):
        st.dataframe(pins, width="stretch", hide_index=True)
        st.caption("Many identical values at the min/max can indicate clipping, saturation, or placeholder "
                   "imputation upstream. Verify with the data source before trusting those rows.")
    if all(mp[k] for k in ("hot_in", "hot_out", "cold_in", "cold_out")):
        d = df
        bad = ((d[mp["hot_in"]] <= d[mp["hot_out"]]) | (d[mp["cold_out"]] <= d[mp["cold_in"]])
               | (d[mp["cold_out"]] > d[mp["hot_in"]]))
        st.metric("Rows violating basic temperature consistency (Th,in ≤ Th,out, Tc,out ≤ Tc,in, or Tc,out > Th,in)",
                  int(bad.sum()))

    st.subheader("Preprocessing configuration")
    st.caption("The original dataset is never modified. Imputation, clipping and scaling are placed inside the "
               "model pipeline and fitted on the **training split only** (no data leakage).")
    x1, x2, x3 = st.columns(3)
    x1.selectbox("Missing values", M.MISSING_OPTS, key="p_missing")
    x2.selectbox("Outliers", M.OUTLIER_OPTS, key="p_outlier")
    x3.checkbox("Drop exact duplicate rows", key="p_dedup")
    cand = [c for c in df.columns if c != mp["target"]]
    cur = [c for c in st.session_state.get("p_features", []) if c in cand]
    st.session_state["p_features"] = cur or [c for c in default_features(df, mp) if c in cand]
    st.multiselect("Model features", cand, key="p_features")
    st.caption("Default features are the mapped engineering columns. Rf is excluded by default (it is derived from U "
               "and a label-dependent baseline). Q, effectiveness and U are derived from temperatures/flows, so "
               "they are correlated with them — keep or drop them deliberately.")
    feats = st.session_state["p_features"]
    st.subheader("Cleaned dataset preview and row counts")
    if not mp["target"]:
        st.info("No label column is mapped, so the supervised-learning dataset cannot be built. "
                "Showing the original data only.")
        return
    c, f = label_values(df, mp)
    if not feats:
        st.warning("Select at least one feature.")
        return
    try:
        d, y, rep = M.prepare_dataset(df, feats, mp["target"], c, f, st.session_state["p_dedup"],
                                      st.session_state["p_missing"])
    except Exception as e:  # noqa: BLE001
        st.error(f"Preprocessing failed: {e}")
        return
    st.dataframe(pd.DataFrame({"step": list(rep), "rows": list(rep.values())}), hide_index=True)
    st.dataframe(d[feats + [mp["target"]]].head(100), width="stretch")
    cc = y.value_counts().rename({0: "Clean (0)", 1: "Fouled (1)"})
    st.write("Class counts:", cc.to_dict())
    if len(y) and cc.min() / cc.max() < 0.5:
        st.warning("Class imbalance detected; models use class weighting and recall/F1 should be emphasised.")
    ml = st.session_state.get("ml")
    if ml:
        st.markdown("**Transformed training matrix (after fitted preprocessing, first 20 rows)**")
        pipe = next(iter(ml["models"].values()))
        names = M.feature_names(pipe.named_steps["pre"], ml["num"], ml["cat"])
        st.dataframe(pd.DataFrame(pipe.named_steps["pre"].transform(ml["Xtr"].head(20)), columns=names))


# ------------------------------------------------------------------ Section 3
def page_engineering():
    st.header("Section 3 · Engineering Performance Analysis")
    df, mp = require_mapping()
    idx = selected_index(df, mp)
    st.caption("Prototype: automotive oil-cooler-based, PHE-like heat-exchanger (hot: water, cold: coolant). "
               "Dominant fouling of interest: CaCO₃ hard-water scaling.")
    with st.expander("Threshold & severity settings (configurable)", expanded=True):
        a, b, c, d = st.columns(4)
        a.number_input("Fouling-resistance threshold", min_value=0.0, step=0.001, format="%.5f", key="p_thr_val")
        b.selectbox("Threshold unit", list(E.RF_UNITS), key="p_thr_unit")
        c.number_input("Severity: Low→Medium at × threshold", min_value=0.01, step=0.5, key="p_sev_med")
        d.number_input("Severity: Medium→High at × threshold", min_value=0.02, step=0.5, key="p_sev_high")
        st.warning("0.043 m²·K/kW (= 4.3×10⁻⁵ m²·K/W) is a **provisional** literature value carried over from earlier "
                   "discussion. I could not verify its original source, units or applicability to this prototype or to "
                   "CaCO₃ scaling. Treat it as an engineering indicator; replace it with a validated value for your "
                   "prototype. Severity multipliers are illustrative defaults, not validated limits.")
    ev = engineering_state(df, mp, idx)
    res, cmpd = ev["res"], ev["cmp"]
    st.subheader(f"Selected row {idx}" + (f" · actual label: {label_text(df, mp, idx)}" if mp["target"] else ""))
    st.caption(ev["ref_desc"] or "No clean reference defined.")
    uu = "W/(m²·K)"
    k = st.columns(4)
    with k[0]:
        kpi("Uclean", fmt(ev["uclean_si"], ".6g"), uu, "clean reference")
    with k[1]:
        kpi("Uf (operating row)", fmt(ev["uf_si"], ".6g"), uu, "evaluated condition")
    with k[2]:
        kpi("Rf = 1/Uf − 1/Uclean", fmt(res.rf_si, ".4e") if res.available else "n/a", "m²·K/W",
            "calculated" if res.available and ev["wf"] != WF_C else "from dataset" if res.available else "unavailable",
            "na" if not res.available else "warn" if res.negative else "")
    with k[3]:
        kpi("Rf (same value)", fmt(res.rf_si * 1e3, ".4g") if res.available else "n/a", "m²·K/kW")
    k = st.columns(4)
    thr_kw = ev["thr_si"] * 1e3
    tone = "na" if cmpd["at_or_above"] is None else "bad" if cmpd["at_or_above"] else "good"
    with k[0]:
        kpi("Selected threshold", fmt(ev["thr_si"], ".4e"), "m²·K/W", f"= {thr_kw:.4g} m²·K/kW")
    with k[1]:
        kpi("Rf / threshold", fmt(cmpd["ratio"], ".3f"), "×", "")
    with k[2]:
        kpi("Threshold status", cmpd["status"], "", "", tone)
    with k[3]:
        ind = ("Cleaning indicated — inspect" if cmpd["at_or_above"] and not res.negative else
               "No cleaning indicated by Rf" if cmpd["at_or_above"] is False and not res.negative else "Not assessable")
        kpi("Cleaning indication", ind, "", "engineering indicator only", tone if not res.negative else "warn")
    if not res.available:
        st.error(f"Rf cannot be calculated reliably: {res.message} A valid clean reference (Workflow A/B) or a "
                 "measured Rf column (Workflow C) is required. No value has been invented.")
    if res.negative:
        st.warning("⚠ Rf is negative. Inspect the baseline, operating conditions, measurement uncertainty and data quality.")
    for n in ev["notes"]:
        st.info(n)
    st.info("The threshold comparison is an **engineering indicator**; it does not independently prove that the "
            "exchanger is clean or fouled. It depends on the baseline, operating point and measurement uncertainty.")

    st.subheader("Operating parameters of the selected row")
    row = df.loc[idx]
    rows = []
    for role in ("hot_in", "hot_out", "cold_in", "cold_out", "hot_flow", "cold_flow", "hot_dp", "cold_dp", "q",
                 "effectiveness", "u", "rf"):
        col = mp[role]
        if col:
            unit = st.session_state["p_u_unit"] if role == "u" else st.session_state["p_rf_unit"] if role == "rf" \
                else role_unit(role, col)
            rows.append({"Parameter": D.ROLES[role][0], "Column": col, "Value": E.to_float(row[col]), "Unit": unit})
    if mp["hot_in"] and mp["hot_out"]:
        v = E.to_float(row[mp["hot_in"]]), E.to_float(row[mp["hot_out"]])
        if None not in v:
            rows.append({"Parameter": "Hot-side ΔT (derived)", "Column": "Th,in − Th,out", "Value": v[0] - v[1],
                         "Unit": role_unit("hot_in", mp["hot_in"])})
    if mp["cold_in"] and mp["cold_out"]:
        v = E.to_float(row[mp["cold_in"]]), E.to_float(row[mp["cold_out"]])
        if None not in v:
            rows.append({"Parameter": "Cold-side ΔT (derived)", "Column": "Tc,out − Tc,in", "Value": v[1] - v[0],
                         "Unit": role_unit("cold_in", mp["cold_in"])})
    rows.append({"Parameter": "Fouling threshold", "Column": "(setting)", "Value": float(st.session_state["p_thr_val"]),
                 "Unit": st.session_state["p_thr_unit"]})
    st.dataframe(pd.DataFrame(rows).assign(Value=lambda t: t["Value"].map(lambda x: fmt(x, ".5g"))),
                 width="stretch", hide_index=True)
    st.caption("Mass-flow-based energy balances are not computed: they need fluid density and specific heat, which "
               "are not in the dataset and are not assumed.")

    g1, g2 = st.columns(2)
    if res.available:
        if ev["uf_si"] is not None and ev["uclean_si"] is not None:
            f1 = go.Figure(go.Bar(x=["Uclean", "Uf"], y=[ev["uclean_si"], ev["uf_si"]],
                                  marker_color=["#2e9e5b", "#d64545"]))
            f1.update_layout(title="Overall heat-transfer coefficient (W/(m²·K))", height=340)
            g1.plotly_chart(f1, width="stretch")
        f2 = go.Figure(go.Bar(x=["Rf (row)", "Threshold"], y=[res.rf_si * 1e3, thr_kw],
                              marker_color=["#d64545" if cmpd["at_or_above"] else "#2e9e5b", "#888"]))
        f2.update_layout(title="Rf vs threshold (m²·K/kW)", height=340)
        g2.plotly_chart(f2, width="stretch")

    if ev["uclean_si"] and mp["u"] and ev["wf"] != WF_C:
        st.subheader("Rf across the dataset (same clean reference)")
        u_si = pd.to_numeric(df[mp["u"]], errors="coerce") * E.U_UNITS[st.session_state["p_u_unit"]]
        rs = E.rf_series(u_si, ev["uclean_si"]) * 1e3
        pdf = pd.DataFrame({"Rf (m²·K/kW)": rs})
        if mp["target"]:
            pdf["label"] = df[mp["target"]].astype(str)
        pdf = pdf.dropna(subset=["Rf (m²·K/kW)"])
        fh = px.histogram(pdf, x="Rf (m²·K/kW)", color="label" if mp["target"] else None, nbins=80,
                          barmode="overlay", opacity=.65, height=360)
        fh.add_vline(x=thr_kw, line_dash="dash", annotation_text="threshold")
        if res.available:
            fh.add_vline(x=res.rf_si * 1e3, line_color="red", annotation_text="selected row")
        st.plotly_chart(fh, width="stretch")
        share = (pdf["Rf (m²·K/kW)"] >= thr_kw)
        tab = (share.groupby(pdf["label"]).mean().mul(100).round(1).rename("% rows ≥ threshold").to_frame()
               if mp["target"] else None)
        if tab is not None:
            st.dataframe(tab)
        st.caption(f"Rows with invalid/missing U ({int(len(df) - len(pdf)):,}) are excluded. If almost all rows of "
                   "both classes lie above the threshold, the threshold is not discriminating for this dataset — "
                   "revisit the threshold/baseline rather than trusting the indicator.")
    with st.expander("Future Rf forecasting (LSTM) — feasibility"):
        if not mp["time"]:
            st.warning("Forecasting is **not implemented** for this dataset: no timestamp/time column is mapped, so "
                       "there is no defensible chronological Rf series. Required: timestamped measurements of one "
                       "exchanger under comparable operating conditions, with a regular sampling interval and enough "
                       "history covering fouling build-up (typically hundreds of points across several fouling cycles), "
                       "plus a stable clean baseline.")
        else:
            t = pd.to_datetime(df[mp["time"]], errors="coerce")
            st.info(f"Time column '{mp['time']}': {int(t.notna().sum())} parseable timestamps. Forecasting is not "
                    "included in this first version; verify ordering, regular sampling and a single continuous run first.")


# ------------------------------------------------------------------ Section 4
def ml_signature(df, mp):
    s = st.session_state
    return (s.get("file_sig"), tuple(s.get("p_features", [])), mp["target"], s.get("p_clean_val"), s.get("p_fouled_val"),
            s["p_missing"], s["p_outlier"], s["p_iqr_k"], s["p_dedup"], s["p_test_size"], s["p_seed"],
            s["p_split"], s["p_trees"], tuple(s["p_models"]))


def run_training(df, mp):
    s = st.session_state
    c, f = label_values(df, mp)
    d, y, rep = M.prepare_dataset(df, s["p_features"], mp["target"], c, f, s["p_dedup"], s["p_missing"])
    feats = list(s["p_features"])
    bad_cat = [x for x in feats if not pd.api.types.is_numeric_dtype(df[x]) and df[x].nunique() > 30]
    if bad_cat:
        feats = [x for x in feats if x not in bad_cat]
        st.warning(f"Dropped high-cardinality categorical features: {bad_cat}")
    tv = df[mp["time"]] if (s["p_split"].startswith("Time") and mp["time"]) else None
    out = M.train_and_evaluate(d, y, feats, s["p_models"], s["p_test_size"], s["p_seed"], s["p_trees"],
                               s["p_missing"], s["p_outlier"], s["p_iqr_k"], tv)
    out["sig"] = ml_signature(df, mp)
    out["prep_report"] = rep
    return out


def page_ml():
    st.header("Section 4 · Fouling Prediction & Explainable AI")
    df, mp = require_mapping()
    idx = selected_index(df, mp)
    if not mp["target"]:
        st.error("No label column is mapped. Supervised classification cannot be trained or evaluated without genuine "
                 "Clean/Fouled labels. (Engineering analysis in Section 3 still works.)")
        st.stop()
    c, f = label_values(df, mp)
    if c == f:
        st.error("Clean and Fouled label values must differ (Section 1).")
        st.stop()
    cur = [x for x in st.session_state.get("p_features", []) if x in df.columns]
    if not cur:
        cur = [x for x in default_features(df, mp) if x != mp["target"]]
    st.session_state["p_features"] = cur
    st.subheader("A. Training configuration")
    a, b, c3, d = st.columns(4)
    a.multiselect("Models", M.MODEL_NAMES, key="p_models")
    b.slider("Test fraction", 0.1, 0.5, step=0.05, key="p_test_size")
    c3.number_input("Random seed", min_value=0, step=1, key="p_seed")
    d.number_input("RF trees", min_value=50, max_value=1000, step=50, key="p_trees")
    split_opts = ["Stratified random"] + (["Time-ordered (latest rows = test)"] if mp["time"] else [])
    if st.session_state["p_split"] not in split_opts:
        st.session_state["p_split"] = split_opts[0]
    st.selectbox("Split strategy", split_opts, key="p_split")
    st.multiselect("Features (also editable in Section 2)", [x for x in df.columns if x != mp["target"]], key="p_features")
    if not mp["time"]:
        st.caption("No timestamp/run-ID column mapped → random stratified split. If rows come from repeated runs of "
                   "the same operating condition, a random split can be optimistic; use a group/time-aware split.")
    if st.button("🚀 Train and evaluate models", type="primary"):
        if not st.session_state["p_models"] or not st.session_state["p_features"]:
            st.error("Select at least one model and one feature.")
        else:
            try:
                with st.spinner("Training models (preprocessing is fitted on the training split only)…"):
                    st.session_state["ml"] = run_training(df, mp)
                    st.session_state["shap_cache"] = {}
            except Exception as e:  # noqa: BLE001
                st.session_state.pop("ml", None)
                st.error(f"Training failed: {e}")
    ml = st.session_state.get("ml")
    if not ml:
        st.info("Train the models to see evaluation, predictions and SHAP explanations.")
        return
    if ml["sig"] != ml_signature(df, mp):
        st.warning("Settings changed since the last training run — retrain to refresh results.")
    for n, e in ml["errors"].items():
        st.warning(e)
    st.caption(f"Split: {ml['split']} · train {ml['n_train']:,} / test {ml['n_test']:,} rows · class counts {ml['class_counts']} "
               "(0 = Clean, 1 = Fouled after your label mapping).")

    st.subheader("B. Model evaluation (held-out test set)")
    met = ml["metrics"].set_index("Model")
    st.dataframe(met.style.format("{:.3f}"), width="stretch")
    long = met.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
    st.plotly_chart(px.bar(long, x="Metric", y="Score", color="Model", barmode="group", height=360),
                    width="stretch")
    cols = st.columns(len(ml["cms"]))
    for col, (name, cm) in zip(cols, ml["cms"].items()):
        fig = px.imshow(cm, text_auto=True, x=["Pred Clean", "Pred Fouled"], y=["Actual Clean", "Actual Fouled"],
                        color_continuous_scale="Blues", title=name, height=300)
        fig.update_coloraxes(showscale=False)
        col.plotly_chart(fig, width="stretch")
    fr = go.Figure()
    for name, (fpr, tpr) in ml["rocs"].items():
        fr.add_scatter(x=fpr, y=tpr, name=name)
    fr.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash"), name="chance")
    fr.update_layout(title="ROC curves", xaxis_title="False-positive rate", yaxis_title="True-positive rate", height=360)
    st.plotly_chart(fr, width="stretch")
    st.info("**Why recall matters:** a false negative is a fouled exchanger predicted as clean, so real fouling is "
            "missed and efficiency losses continue unnoticed. Compare recall (and the false-negative cell) as well as "
            "accuracy. Metrics only describe this dataset and split; they are not proof of performance on the physical prototype.")

    st.subheader("C. Selected-row prediction")
    row = df.loc[[idx], ml["features"]]
    pr = M.predict_row(ml["models"], row)
    lab = label_text(df, mp, idx)
    where = "training set" if idx in ml["train_idx"] else "test set" if idx in ml["test_idx"] else "excluded from training/testing"
    k = st.columns(3)
    with k[0]:
        kpi("Row identifier", str(idx), "", f"CSV line {idx + 2}")
    with k[1]:
        kpi("Actual label (dataset)", lab or "n/a", "", "ground truth, not a prediction")
    with k[2]:
        kpi("Row used as", where, "", "in-sample predictions are optimistic" if where == "training set" else "")
    if row.isna().any(axis=1).iloc[0]:
        st.warning("This row has missing feature values; they were imputed with training-split statistics for prediction.")
    st.dataframe(pr.style.format({"P(Fouled)": "{:.3f}", "Confidence": "{:.3f}"}), width="stretch", hide_index=True)
    preds = pr["Predicted"].tolist()
    if len(set(preds)) == 1:
        st.success(f"All {len(preds)} models agree: **{preds[0]}** (model prediction).")
    else:
        st.warning("Models disagree for this row: " + ", ".join(f"{m}: {p}" for m, p in zip(pr['Model'], preds)))
    st.plotly_chart(px.bar(pr, x="Model", y="P(Fouled)", range_y=[0, 1], height=300,
                           title="Predicted probability of Fouled (model output, not physical severity)"),
                    width="stretch")

    st.subheader("D. Fouling severity (Rf-based)")
    ev = engineering_state(df, mp, idx)
    sev = ev["sev"]
    thr_kw = ev["thr_si"] * 1e3
    st.markdown(f"Boundaries (configurable in Section 3): **Low** < {sev['low_max'] * 1e3:.4g} · **Medium** "
                f"{sev['low_max'] * 1e3:.4g} – {sev['med_max'] * 1e3:.4g} · **High** ≥ {sev['med_max'] * 1e3:.4g} m²·K/kW")
    if sev["level"]:
        kpi("Severity from Rf", sev["level"], "", f"Rf = {fmt(ev['res'].rf_si * 1e3, '.4g')} m²·K/kW",
            {"Low": "good", "Medium": "warn", "High": "bad"}[sev["level"]])
    else:
        st.warning(f"Severity not shown: {sev['reason'] or ev['res'].message}")
    st.caption("Severity is derived from physical Rf and user-set boundaries (illustrative defaults, not validated). "
               "It is not the classifier's probability, which expresses model confidence, not fouling magnitude.")

    st.subheader("E. SHAP explainability")
    shap_section(df, mp, idx, ml)


def shap_section(df, mp, idx, ml):
    names_ok = list(ml["models"])
    st.selectbox("Model to explain", names_ok, key="p_shap_model")
    if st.session_state.get("p_shap_model") not in names_ok:
        st.session_state["p_shap_model"] = names_ok[0]
    name = st.session_state["p_shap_model"]
    pipe = ml["models"][name]
    cache = st.session_state.setdefault("shap_cache", {})
    try:
        import shap  # noqa: F401
    except Exception as e:  # noqa: BLE001
        st.warning(f"SHAP is not installed/importable ({e}). The rest of the dashboard still works.")
        return
    try:
        if name not in cache:
            bg = ml["Xtr"].sample(min(100, len(ml["Xtr"])), random_state=0)
            cache[name] = {"explainer": X.make_explainer(pipe, name, bg)}
        expl = cache[name]["explainer"]
        raw = df.loc[[idx], ml["features"]]
        sv, xt, names = X.shap_values(pipe, name, expl, raw, ml["num"], ml["cat"])
        lt = X.local_table(sv[0], xt[0], names, df.loc[idx, ml["features"]], {k: v for k, v in
                           ((mp[r], role_unit(r, mp[r])) for r in D.ROLES if mp.get(r) and r not in ("target", "time", "u", "rf")
                            )} | ({mp["u"]: st.session_state["p_u_unit"]} if mp["u"] else {}))
    except Exception as e:  # noqa: BLE001
        st.warning(f"SHAP failed for {name}: {e}. Try another model; the rest of the dashboard remains functional.")
        return
    st.markdown("**Local explanation — selected row**")
    top = lt.head(10).iloc[::-1]
    fig = go.Figure(go.Bar(x=top["SHAP value"], y=top["Feature"], orientation="h",
                           marker_color=["#d64545" if v > 0 else "#2e7bd6" for v in top["SHAP value"]]))
    fig.update_layout(height=380, xaxis_title="SHAP value (→ Fouled, ← Clean; model-output units)",
                      title=f"{name}: feature contributions for row {idx}")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(lt.style.format({"SHAP value": "{:+.4f}"}), width="stretch", hide_index=True)
    st.session_state["local_shap"] = {"model": name, "row": idx, "table": lt}

    st.markdown("**Global explanation — test-set sample**")
    n = st.slider("Rows used for global SHAP", 100, int(min(2000, max(len(ml["Xte"]), 100))), key="p_shap_n") \
        if len(ml["Xte"]) > 100 else len(ml["Xte"])
    if st.button("Compute global SHAP"):
        try:
            with st.spinner("Computing SHAP values…"):
                samp = ml["Xte"].sample(min(n, len(ml["Xte"])), random_state=0)
                gsv, gxt, gnames = X.shap_values(pipe, name, expl, samp, ml["num"], ml["cat"])
                cache[name]["global"] = (gsv, gxt, gnames)
        except Exception as e:  # noqa: BLE001
            st.warning(f"Global SHAP failed: {e}")
    g = cache[name].get("global")
    if g:
        gsv, gxt, gnames = g
        imp = pd.DataFrame({"Feature": gnames, "mean |SHAP|": np.abs(gsv).mean(axis=0)}).sort_values("mean |SHAP|")
        st.plotly_chart(px.bar(imp, x="mean |SHAP|", y="Feature", orientation="h", height=380,
                               title=f"Global feature importance — {name}"), width="stretch")
        try:
            import shap
            plt.figure()
            shap.summary_plot(gsv, gxt, feature_names=gnames, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
        except Exception as e:  # noqa: BLE001
            st.warning(f"SHAP summary plot unavailable: {e}")
    st.caption("SHAP describes how the trained model arrives at its output; it does not by itself establish physical "
               "causation. Positive values push toward Fouled (label 1), negative toward Clean (label 0).")


# ------------------------------------------------------------------ Section 5
def page_maintenance():
    st.header("Section 5 · Maintenance Support")
    df, mp = require_mapping()
    idx = selected_index(df, mp)
    ev = engineering_state(df, mp, idx)
    res, cmpd = ev["res"], ev["cmp"]
    ml = st.session_state.get("ml")
    ml_label, agree, pr = None, None, None
    if ml and ml["sig"] == ml_signature(df, mp):
        pr = M.predict_row(ml["models"], df.loc[[idx], ml["features"]])
        names = pr["Model"].tolist()
        st.session_state.setdefault("p_primary", names[0])
        if st.session_state["p_primary"] not in names:
            st.session_state["p_primary"] = names[0]
        st.selectbox("Primary model for the decision logic", names, key="p_primary")
        ml_label = pr.set_index("Model").loc[st.session_state["p_primary"], "Predicted"]
        if ml_label not in ("Clean", "Fouled"):
            ml_label = None
        agree = pr["Predicted"].nunique() == 1
    elif ml:
        st.warning("Settings changed since training; retrain in Section 4. Showing the engineering-only assessment.")
    else:
        st.info("No trained model yet (Section 4). Showing the engineering-only assessment.")

    top_f, top_c = [], []
    ls = st.session_state.get("local_shap")
    if ml_label and ls and ls["row"] == idx:
        rmap = {c: r for r, c in mp.items() if c}
        t = ls["table"]
        fmtv = lambda r: f"{fmt(r['Actual value'], '.4g')} {r['Unit']}".strip()  # noqa: E731
        top_f = [(r["Feature"], rmap.get(r["Feature"]), fmtv(r)) for _, r in t[t["SHAP value"] > 0].head(3).iterrows()]
        top_c = [(r["Feature"], rmap.get(r["Feature"]), fmtv(r)) for _, r in t[t["SHAP value"] < 0].head(3).iterrows()]
        shap_model = ls["model"]
    else:
        shap_model = None

    dq = []
    row = df.loc[idx]
    mapped_cols = [c for r, c in mp.items() if c and r not in ("target", "time")]
    miss = [c for c in mapped_cols if pd.isna(row[c])]
    if miss:
        dq.append(f"Selected row has missing values in: {', '.join(miss)}.")
    for c in mapped_cols:
        s = df[c].dropna()
        if len(s) and pd.notna(row[c]) and row[c] in (s.min(), s.max()) and (s == row[c]).sum() > 0.005 * len(s):
            dq.append(f"{c} = {row[c]:.4g} equals the column {'minimum' if row[c] == s.min() else 'maximum'} shared by "
                      f"{int((s == row[c]).sum()):,} rows (possible clipping/placeholder value).")
    if mp["hot_in"] and mp["hot_out"] and pd.notna(row[mp["hot_in"]]) and pd.notna(row[mp["hot_out"]]) \
            and row[mp["hot_out"]] >= row[mp["hot_in"]]:
        dq.append("Hot outlet temperature ≥ hot inlet temperature (physically inconsistent).")
    if mp["cold_in"] and mp["cold_out"] and pd.notna(row[mp["cold_in"]]) and pd.notna(row[mp["cold_out"]]) \
            and row[mp["cold_out"]] <= row[mp["cold_in"]]:
        dq.append("Cold outlet temperature ≤ cold inlet temperature (physically inconsistent).")
    if ml and idx in ml["train_idx"]:
        dq.append("This row was part of the model training set, so its prediction is in-sample.")
    if ev["wf"] == WF_A and st.session_state["p_ref_mode"] == REF_ONE:
        dq.append("Clean baseline is a single row; measurement noise in that row propagates directly into Rf.")
    if df.isna().mean().max() > 0.02:
        dq.append(f"Dataset has up to {df.isna().mean().max() * 100:.1f}% missing values per column.")
    dq += [n for n in ev["notes"] if n not in dq]

    roles_cols = {r: c for r, c in mp.items() if c}
    cl = label_values(df, mp)[0]
    opn = R.operating_notes(df, row, roles_cols, mp["target"], cl)
    out = R.decide(ml_label, res.available, res.negative, cmpd["at_or_above"], cmpd["ratio"], ev["sev"]["level"],
                   agree, top_f, top_c, dq, opn)

    k = st.columns(4)
    with k[0]:
        kpi("Row", str(idx), "", f"actual label: {label_text(df, mp, idx) or 'n/a'}")
    with k[1]:
        kpi("Rf / threshold", fmt(cmpd["ratio"], ".2f"), "×", cmpd["status"],
            "na" if cmpd["at_or_above"] is None else "bad" if cmpd["at_or_above"] else "good")
    with k[2]:
        kpi("ML prediction", ml_label or "n/a", "", f"model: {st.session_state.get('p_primary', '—')}" if ml_label else "",
            "na" if not ml_label else "bad" if ml_label == "Fouled" else "good")
    with k[3]:
        kpi("Severity (Rf-based)", ev["sev"]["level"] or "n/a", "", "configurable boundaries")
    box = {"error": st.error, "warning": st.warning, "success": st.success, "info": st.info}[out["level"]]
    box(f"**{out['headline']}**" + (f"  \nDecision case {out['case']}" if out["case"] else ""))
    st.subheader("Recommended actions")
    for a in out["actions"]:
        st.markdown(f"- {a}")
    if pr is not None:
        st.markdown("**Model predictions for this row**")
        st.dataframe(pr.style.format({"P(Fouled)": "{:.3f}", "Confidence": "{:.3f}"}), hide_index=True)
    if top_f or top_c:
        st.subheader(f"SHAP evidence ({shap_model})")
        c1, c2 = st.columns(2)
        c1.markdown("**Pushing toward Fouled**")
        for f_, r_, v_ in top_f:
            c1.markdown(f"- {f_} = {v_}")
        c2.markdown("**Pushing toward Clean**")
        for f_, r_, v_ in top_c:
            c2.markdown(f"- {f_} = {v_}")
        for h in out["hints"]:
            st.markdown(f"> {h}")
    elif ml_label:
        st.info("Open Section 4 · SHAP for this row to add explanation evidence here.")
    if opn:
        st.subheader("Comparison with Clean-labelled rows")
        for n in opn:
            st.markdown(f"- {n}")
    if dq:
        st.subheader("Data-quality warnings")
        for w in dq:
            st.markdown(f"- ⚠ {w}")
    st.caption(R.DISCLAIMER)
    st.caption("SHAP describes model behaviour, not physical causation; the Rf threshold is an engineering indicator, "
               "not proof of cleanliness or fouling.")


# ------------------------------------------------------------------ main
st.sidebar.title("🔧 HX Fouling Dashboard")
st.sidebar.caption("Automotive oil-cooler-based, PHE-like prototype · hot: water · cold: coolant · CaCO₃ scaling focus")
st.sidebar.radio("Section", SECTIONS, key="p_section")
if st.session_state.get("df") is not None:
    st.sidebar.success(f"Dataset: {st.session_state['file_name']} ({len(st.session_state['df']):,} rows)")
    if st.session_state.get("p_confirmed"):
        _df, _mp = st.session_state["df"], mapping()
        try:
            st.sidebar.info(f"Selected row: {selected_index(_df, _mp)}")
        except Exception:  # noqa: BLE001
            pass
PAGES = dict(zip(SECTIONS, [page_input, page_preprocess, page_engineering, page_ml, page_maintenance]))
try:
    PAGES[st.session_state["p_section"]]()
except Exception as _e:  # noqa: BLE001
    if type(_e).__name__ in ("StopException", "RerunException"):
        raise
    st.error(f"Unexpected error: {_e}")
    st.exception(_e)
