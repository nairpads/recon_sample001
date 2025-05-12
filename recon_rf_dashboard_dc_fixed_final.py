
import streamlit as st
import pandas as pd
import numpy as np
import joblib

def apply_logical_override_features(pairs_df):
    pairs_df['pre_override'] = ((pairs_df['amount_diff'] <= 1) &
                                (pairs_df['date_diff'] <= 2) &
                                (pairs_df['dc_mirror'] == 1))
    return pairs_df

def apply_post_match_override(row):
    if row.get('amount_diff', 999) == 0 and row.get('date_diff', 999) == 0 and row.get('dc_mirror', 0) == 1:
        row['match'] = 1
        row['confidence'] = max(row.get('confidence', 0), 0.99)
    return row

st.title("🧾 Reconciliation Match Predictor")
st.markdown("Upload your MT940 and Ledger files. We'll suggest potential matches using logic and ML.")

mt_file = st.file_uploader("Upload MT940 Parsed File (CSV)", type=["csv"])
ledger_file = st.file_uploader("Upload Ledger File (CSV)", type=["csv"])
model_file = st.file_uploader("Upload Trained Model (.pkl)", type=["pkl"])

if mt_file and ledger_file and model_file:
    mt_df = pd.read_csv(mt_file)
    ledger_df = pd.read_csv(ledger_file)
    model = joblib.load(model_file)

    pairs_df = pd.merge(mt_df.reset_index(), ledger_df.reset_index(), how='cross', suffixes=('_mt', '_ld'))
    pairs_df['amount_diff'] = (pairs_df['amount_mt'] - pairs_df['amount_ld']).abs()
    pairs_df['date_diff'] = (pd.to_datetime(pairs_df['date_mt']) - pd.to_datetime(pairs_df['date_ld'])).dt.days.abs()

    dc_mt_col = next((col for col in pairs_df.columns if 'dc' in col and col.endswith('_mt')), None)
    dc_ld_col = next((col for col in pairs_df.columns if 'dr_cr' in col and col.endswith('_ld')), None)
    if dc_mt_col and dc_ld_col:
        mt_dc = pairs_df[dc_mt_col].astype(str).str.upper().str.strip()
        ld_dc = pairs_df[dc_ld_col].astype(str).str.upper().str.strip()
        pairs_df['dc_mirror'] = (((mt_dc == 'D') & (ld_dc == 'C')) | ((mt_dc == 'C') & (ld_dc == 'D'))).astype(int)
    else:
        pairs_df['dc_mirror'] = 0

    ref_mt_col = next((col for col in pairs_df.columns if 'ref' in col and col.endswith('_mt')), None)
    ref_ld_col = next((col for col in pairs_df.columns if 'ref' in col and col.endswith('_ld')), None)
    if ref_mt_col and ref_ld_col:
        mt_ref = pairs_df[ref_mt_col].astype(str).str.lower().str.strip()
        ld_ref = pairs_df[ref_ld_col].astype(str).str.lower().str.strip()
        pairs_df['ref_match'] = (mt_ref == ld_ref).astype(int)
    else:
        pairs_df['ref_match'] = 0

    pairs_df['desc_match'] = pairs_df.apply(
        lambda x: 1 if str(x.get('narration_mt', '')).lower() in str(x.get('narration_ld', '')).lower()
        or str(x.get('narration_ld', '')).lower() in str(x.get('narration_mt', '')).lower() else 0,
        axis=1)

    pairs_df = apply_logical_override_features(pairs_df)

    X = pairs_df[['amount_diff', 'date_diff', 'dc_mirror', 'ref_match', 'desc_match']]
    pairs_df['match'] = model.predict(X)
    pairs_df['confidence'] = model.predict_proba(X)[:, 1]

    pairs_df = pairs_df.apply(apply_post_match_override, axis=1)

    csv = pairs_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Reconciliation Results", csv, "recon_rf_output.csv", "text/csv")
    st.dataframe(pairs_df.head(50))
