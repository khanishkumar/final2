"""Transparent rule-based maintenance decision support."""
from __future__ import annotations
import numpy as np
import pandas as pd

DISCLAIMER = ("Decision-support guidance only. It does not replace engineering inspection. Do not select "
              "cleaning chemicals, concentrations or durations without verified material compatibility "
              "and the manufacturer's cleaning requirements.")

ROLE_HINTS = {
    "hot_dp": "Higher pressure drop can accompany deposit build-up or flow restriction; verify the ΔP sensor and flow rate.",
    "cold_dp": "Higher pressure drop can accompany deposit build-up or flow restriction; verify the ΔP sensor and flow rate.",
    "u": "A low U relative to the clean reference indicates degraded heat transfer; confirm the baseline is representative.",
    "hot_flow": "Flow differences change U and ΔP independently of fouling; confirm the operating point matches the baseline.",
    "cold_flow": "Flow differences change U and ΔP independently of fouling; confirm the operating point matches the baseline.",
    "hot_in": "Check temperature-sensor calibration and operating-point differences.",
    "hot_out": "Check temperature-sensor calibration and operating-point differences.",
    "cold_in": "Check temperature-sensor calibration and operating-point differences.",
    "cold_out": "Check temperature-sensor calibration and operating-point differences.",
    "q": "Heat duty is derived from temperatures/flows; confirm those inputs before trusting it.",
}


def operating_notes(df, row, roles_cols: dict, target, clean_val):
    """Compare selected-row values with the median of Clean-labelled rows (data-driven, no assumptions)."""
    notes = []
    if not target or clean_val is None:
        return notes
    clean = df[df[target].astype(str) == str(clean_val)]
    if clean.empty:
        return notes
    for role in ("hot_dp", "cold_dp", "u"):
        col = roles_cols.get(role)
        if not col or col not in row.index or pd.isna(row[col]):
            continue
        med = clean[col].median()
        if med and np.isfinite(med) and med != 0:
            notes.append(f"{col} = {row[col]:.4g} is {row[col] / med:.2f}× the median of Clean-labelled rows ({med:.4g}).")
    return notes


def decide(ml_label, rf_available, rf_negative, at_or_above, ratio, severity_level, models_agree,
           top_fouled, top_clean, dq_warnings, op_notes):
    """ml_label: 'Clean'|'Fouled'|None.  top_*: list of (feature, role, value_text)."""
    rf_ok = bool(rf_available) and not rf_negative
    acts, case, level, head = [], "", "info", ""
    if ml_label is None:
        case, head = "ML unavailable", "Engineering-only assessment (no trained model)"
        if rf_ok and at_or_above:
            level = "warning"
            acts = ["Rf is at/above the selected threshold: verify measurements and inspect the exchanger.",
                    "Investigate possible mineral (e.g. CaCO₃) scaling and consider a compatible cleaning method."]
        elif rf_ok:
            level = "success"
            acts = ["Rf is below the selected threshold: continue routine monitoring and record performance trends."]
        else:
            acts = ["Rf is not reliably available: a valid clean baseline or measured Rf is required."]
    elif not rf_ok:
        case, level = "5 · Rf unavailable/unreliable", "info"
        head = f"ML result: {ml_label}. Engineering confirmation is not possible."
        acts = ["Treat the ML result as an indication only.",
                "Engineering confirmation requires a valid clean baseline (Uclean) or a measured Rf.",
                "If Rf was negative, inspect the baseline, operating conditions, measurement uncertainty and data quality."]
    elif ml_label == "Fouled" and at_or_above:
        case, level, head = "1 · ML Fouled & Rf ≥ threshold", "error", "Fouling indicated by both ML and engineering analysis"
        acts = ["Verify the measurements (sensors, flow rates) used for this row.",
                "Inspect the heat exchanger for deposits.",
                "Investigate possible mineral scaling (e.g. CaCO₃), considering water hardness and operating temperatures.",
                "Consider a cleaning method compatible with the exchanger's materials, following manufacturer guidance."]
    elif ml_label == "Fouled":
        case, level, head = "2 · ML Fouled & Rf < threshold", "warning", "ML indicates fouling, but Rf is below the threshold"
        acts = ["Investigate the most influential parameters (see SHAP list below).",
                "Check that the clean baseline (Uclean) is representative of the same operating conditions.",
                "Review data quality for this row (missing values, boundary-pinned values, sensor faults)."]
    elif at_or_above:
        case, level, head = "3 · ML Clean & Rf ≥ threshold", "warning", "Disagreement: ML says Clean but Rf is at/above the threshold"
        acts = ["Do not ignore the engineering warning.",
                "Verify the clean baseline and the measurements behind Uf.",
                "Inspect the system when appropriate; the model may not capture this operating condition."]
    else:
        case, level, head = "4 · ML Clean & Rf < threshold", "success", "No fouling indicated by ML or engineering analysis"
        acts = ["Continue routine monitoring.", "Record performance trends (U, Rf, pressure drop) over time."]
    if ratio is not None and rf_ok and ratio >= 1:
        acts.append(f"Rf is {ratio:.2f}× the selected threshold.")
    if severity_level:
        acts.append(f"Rf-based severity (configurable boundaries): {severity_level}.")
    if models_agree is False:
        acts.append("Trained models disagree for this row: treat the ML result with extra caution.")
    hints = []
    for feat, role, val in top_fouled[:3]:
        if role in ROLE_HINTS and ROLE_HINTS[role] not in hints:
            hints.append(ROLE_HINTS[role])
    return {"case": case, "level": level, "headline": head, "actions": acts, "hints": hints,
            "top_fouled": top_fouled, "top_clean": top_clean, "dq": dq_warnings, "op_notes": op_notes}
