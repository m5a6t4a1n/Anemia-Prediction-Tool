# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap

from feature_config import (
    CONT_FEATURES, CAT_FEATURES, ALL_FEATURES, FEATURE_LABELS,
    FEATURE_HELP, SIDEBAR_FEATURE_DESC,
    CONT_INPUT_SPEC, CAT_LEVELS, REGION_PROVINCES,
    MODEL_PATH, BACKGROUND_PATH,
)

st.set_page_config(
    page_title="Incident Anemia Risk Prediction Tool",
    layout="wide",
)


# ----------------------------------------------------------------------
# Asset loading (cached so repeated interactions do not reload)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_assets():
    pipe = joblib.load(MODEL_PATH)
    background = joblib.load(BACKGROUND_PATH)
    # One TreeExplainer for each imputed model.
    explainer_list = [
        shap.TreeExplainer(pipe.model_member(k))
        for k in range(pipe.n_imputations)
    ]
    return pipe, background, explainer_list


def encoded_names_for(pipe, idx):
    """Encoded feature names consistent with training CELL 58 (no CT prefix)."""
    pre = pipe.preprocessor_member(idx)
    cat_names = pre.named_transformers_["cat"].get_feature_names_out(CAT_FEATURES)
    return list(CONT_FEATURES) + list(cat_names)


def member_original_shap(pipe, explainer, idx, x_df):
    """Single set: encoded-space SHAP -> aggregated to the eight original variables."""
    x_enc = pipe.transform_member(x_df)
    sv = explainer(x_enc)

    if sv.values.ndim == 3:          # (n, n_enc, 2)
        vals = sv.values[..., 1]
        base = np.atleast_1d(explainer.expected_value)[1]
    else:
        vals = sv.values
        base = np.atleast_1d(explainer.expected_value)[0]

    names = encoded_names_for(pipe, idx)
    orig = np.zeros((len(x_df), len(ALL_FEATURES)))

    # Continuous variables: map directly by name.
    for i, feat in enumerate(CONT_FEATURES):
        orig[:, i] = vals[:, names.index(feat)]

    # Categorical variables: SHAP of the dummy for the active category
    # (reference categories have no dummy and are recorded as 0).
    for j, feat in enumerate(CAT_FEATURES):
        col = len(CONT_FEATURES) + j
        dummies = [n for n in names if n.startswith(feat + "_")]
        for s in range(len(x_df)):
            dummy_name = f"{feat}_{int(x_df.iloc[s][feat])}"
            if dummy_name in dummies:
                orig[s, col] = vals[s, names.index(dummy_name)]
    return orig, base


def pooled_original_shap(pipe, explainer_list, x_df):
    """Average the original-variable SHAP values and base values across five sets."""
    per_shap, per_base = [], []
    for k, explainer in enumerate(explainer_list):
        o, b = member_original_shap(pipe, explainer, k, x_df)
        per_shap.append(o)
        per_base.append(b)
    return np.mean(per_shap, axis=0), float(np.mean(per_base))


def build_input_dataframe(values):
    return pd.DataFrame([values], columns=ALL_FEATURES)


# ----------------------------------------------------------------------
# Sidebar: tool information (algorithm + feature information)
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("About this tool")

    st.markdown("**Algorithm**: LightGBM gradient boosting model.")
    st.markdown("**Feature information**: 8 feature variables.")

    for feat in ALL_FEATURES:
        st.markdown(f"**{FEATURE_LABELS[feat]}**: {SIDEBAR_FEATURE_DESC[feat]}")
        if feat == "Region":
            for code, (name, provs) in REGION_PROVINCES.items():
                st.markdown(f"　{code} {name}: {', '.join(provs)}")

    st.caption("Disclaimer: the output is a probability estimate based on population data "
               "and does not constitute a diagnosis or treatment recommendation. "
               "Anemia is confirmed by hemoglobin testing.")


# ----------------------------------------------------------------------
# Main page: title + eight inputs (four rows, two columns)
# ----------------------------------------------------------------------
st.title("4-Year Incident Anemia Risk Prediction Tool for Older Chinese Adults")
st.caption(
    "Based on the CHARLS longitudinal cohort and a LightGBM model, using eight "
    "non-blood variables accessible in community and home settings. This tool is "
    "intended for early risk stratification and health education, and does not "
    "replace clinical diagnosis."
)

try:
    pipe, background, explainer_list = load_assets()
except Exception as e:
    st.error(
        "Model files not found. Please run build_and_export in export_model.py "
        "in the training notebook to generate the joblib files first."
    )
    st.exception(e)
    st.stop()

inputs = {}
grid = st.columns(2)
for i, feat in enumerate(ALL_FEATURES):
    with grid[i % 2]:
        if feat in CONT_FEATURES:
            lo, hi, default, step = CONT_INPUT_SPEC[feat]
            val = st.number_input(
                FEATURE_LABELS[feat],
                min_value=lo, max_value=hi, value=default, step=step,
                help=FEATURE_HELP[feat],
            )
            inputs[feat] = val
        else:
            levels = CAT_LEVELS[feat]
            chosen = st.selectbox(
                FEATURE_LABELS[feat], options=list(levels.keys()),
                format_func=lambda k, lv=levels: lv[k],
                help=FEATURE_HELP[feat],
            )
            inputs[feat] = chosen

run = st.button("Run prediction", type="primary")

# Results
if run:
    x_df = build_input_dataframe(inputs)
    prob = float(pipe.predict_proba(x_df)[:, 1][0])
    st.metric("Predicted 4-year probability of incident anemia", f"{prob * 100:.1f}%")

    st.divider()
    st.subheader("Individual explanation for this prediction (SHAP force plot)")
    st.write("Red factors increase the risk and blue factors reduce it; "
             "longer bars indicate larger contributions.")

    orig_shap, base_val = pooled_original_shap(pipe, explainer_list, x_df)

    # Values shown in the force plot: continuous values to two decimals,
    # categorical values kept as integers.
    disp_row = x_df[ALL_FEATURES].iloc[0].copy()
    for f in CONT_FEATURES:
        disp_row[f] = round(float(disp_row[f]), 2)

    force_exp = shap.Explanation(
        values=orig_shap[0],
        base_values=base_val,
        data=disp_row.to_numpy(),
        feature_names=ALL_FEATURES,
    )
    plt.figure(figsize=(12, 4))
    shap.plots.force(force_exp, matplotlib=True, show=False)
    st.pyplot(plt.gcf())
    plt.close("all")

    with st.expander("View the values entered for this prediction"):
        show = {}
        for f in ALL_FEATURES:
            if f in CAT_FEATURES:
                show[FEATURE_LABELS[f]] = CAT_LEVELS[f][inputs[f]]
            else:
                show[FEATURE_LABELS[f]] = inputs[f]
        st.dataframe(pd.DataFrame([show]), width="stretch")

# Global explanation (always shown; five-set aggregated SHAP on background data).
st.divider()
st.subheader("Global feature importance (SHAP, based on background samples)")
bg_shap, _ = pooled_original_shap(pipe, explainer_list, background[ALL_FEATURES])
bg_exp = shap.Explanation(values=bg_shap, feature_names=ALL_FEATURES)
fig, ax = plt.subplots(figsize=(9, 6))
shap.plots.bar(bg_exp, show=False, ax=ax)
st.pyplot(fig)
plt.close(fig)
