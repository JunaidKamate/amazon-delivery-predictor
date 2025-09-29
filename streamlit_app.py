# streamlit_app.py — polished Streamlit app (ready-to-run)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta

import os

# ---------- CONFIG (relative paths for Streamlit Cloud) ----------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models_and_results", "RandomForest_model_no_leak.joblib")
ENGINEERED_CSV = os.path.join(BASE_DIR, "amazon_delivery_step7_engineered.csv")
FI_CSV = os.path.join(BASE_DIR, "feature_importance", "RandomForest_permutation_importances.csv")
# ----------------------------------------------------


# Default fallbacks
DEFAULT_CATEGORY_LIST = ["Electronics", "Grocery", "Clothing", "Books", "Toys", "Home"]
DEFAULT_CATEGORY_FREQ = 0.01
DEFAULT_ORDER_TO_PICKUP_MIN = 15.0

st.set_page_config(page_title="Amazon Delivery Time Predictor", layout="centered")
st.title("Amazon Delivery Time Predictor — Prototype")
st.markdown(
    "Enter order details (left) and click **Predict delivery time**. "
    "This is a demonstration prototype — for production add auth, validation, logging, and tests."
)

# --------- Utility: load model & mapping ---------
@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None, f"Model file not found: {path}"
    try:
        m = joblib.load(path)
        return m, None
    except Exception as e:
        return None, f"Failed to load model: {e}"

@st.cache_data
def load_category_freq(csv_path: str):
    """
    Returns a dict: {category_label: category_freq}
    If file or columns missing, returns empty dict.
    """
    if not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path)
        if "Category" in df.columns and "category_freq" in df.columns:
            # Use the mean category_freq per Category if duplicates exist
            return df.groupby("Category")["category_freq"].mean().to_dict()
    except Exception:
        pass
    return {}

# load assets
model, model_load_err = load_model(MODEL_PATH)
category_freq_map = load_category_freq(ENGINEERED_CSV)

# --------- Rename maps (fix typos from raw data) ---------
# Add any other known typos here
rename_map_category = {
    "Metropolitian": "Metropolitan",
    # add more category typos if you discover them
}
rename_map_area = {
    "Metropolitian": "Metropolitan",
    # add more area typos if needed
}

# Normalize category keys used in UI: map training keys through rename_map
def normalized_category_options(cat_map: dict):
    if not cat_map:
        return DEFAULT_CATEGORY_LIST
    # apply rename_map to keys and keep mapping from normalized -> original key (if needed)
    normalized = []
    for k in sorted(cat_map.keys()):
        nk = rename_map_category.get(k, k)
        normalized.append(nk)
    # remove duplicates while preserving order
    seen = set()
    normalized_unique = []
    for v in normalized:
        if v not in seen:
            normalized_unique.append(v)
            seen.add(v)
    return normalized_unique

ui_category_options = normalized_category_options(category_freq_map)

# --------- UI Inputs ---------
st.subheader("Order & Delivery Details")
col1, col2 = st.columns(2)

with col1:
    distance_km = st.number_input("Distance (km)", min_value=0.0, value=5.0, step=0.1, format="%.2f")
    order_date = st.date_input("Order date", value=datetime.today())
    order_time = st.time_input("Order time", value=datetime.now().time())
    order_dt = datetime.combine(order_date, order_time)
    category = st.selectbox("Product Category", options=ui_category_options)

with col2:
    agent_age = st.number_input("Agent age (years)", min_value=18, max_value=100, value=30)
    agent_rating = st.number_input("Agent rating (1.0 - 5.0)", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
    have_pickup = st.checkbox("I have pickup time to calculate order → pickup gap")
    if have_pickup:
        pickup_date = st.date_input("Pickup date", value=(datetime.today()))
        pickup_time = st.time_input("Pickup time", value=(datetime.now() + timedelta(minutes=15)).time())
        pickup_dt_val = datetime.combine(pickup_date, pickup_time)
    else:
        pickup_dt_val = None

# Show model status
if model is None:
    st.warning("Model not loaded. Check MODEL_PATH. " + (model_load_err or ""))
else:
    st.info("Model loaded and ready.")

submitted = st.button("Predict delivery time")

# --------- Helpers ---------
def map_category_to_training_key(selected_cat: str, train_map: dict):
    """
    Training map has original keys (possibly with typos).
    We normalized options for UI. Map UI selection back to training key:
    - If selected_cat matches a normalized training key, return original key.
    - Else, return selected_cat (will use fallback frequency).
    """
    if not train_map:
        return None
    # Build reverse mapping from normalized -> original
    rev = {}
    for orig in train_map.keys():
        norm = rename_map_category.get(orig, orig)
        rev[norm] = orig
    return rev.get(selected_cat, None)

def safe_category_freq(cat_selected: str):
    train_key = map_category_to_training_key(cat_selected, category_freq_map)
    if train_key and train_key in category_freq_map:
        return float(category_freq_map[train_key])
    # fallback
    return float(DEFAULT_CATEGORY_FREQ)

def compute_features(distance_km, order_dt, agent_age, agent_rating, category_selected, pickup_dt_val):
    # time features
    order_dt = pd.to_datetime(order_dt)
    order_hour = int(order_dt.hour)
    order_dayofweek = int(order_dt.dayofweek)
    order_month = int(order_dt.month)
    order_is_weekend = int(order_dayofweek in [5, 6])

    # order_to_pickup_min
    if pickup_dt_val is not None:
        try:
            pickup_dt_parsed = pd.to_datetime(pickup_dt_val)
            gap_min = (pickup_dt_parsed - order_dt).total_seconds() / 60.0
            if np.isnan(gap_min) or gap_min < 0:
                order_to_pickup_min = DEFAULT_ORDER_TO_PICKUP_MIN
            else:
                order_to_pickup_min = gap_min
        except Exception:
            order_to_pickup_min = DEFAULT_ORDER_TO_PICKUP_MIN
    else:
        order_to_pickup_min = DEFAULT_ORDER_TO_PICKUP_MIN

    cat_freq = safe_category_freq(category_selected)

    features = {
        "Agent_Age": float(agent_age),
        "Agent_Rating": float(agent_rating),
        "order_to_pickup_min": float(order_to_pickup_min),
        "distance_km": float(distance_km),
        "order_hour": int(order_hour),
        "order_dayofweek": int(order_dayofweek),
        "order_month": int(order_month),
        "order_is_weekend": int(order_is_weekend),
        "category_freq": float(cat_freq)
    }
    return pd.DataFrame([features])

def rf_uncertainty_if_rf(pipeline, X_df):
    """
    If pipeline is sklearn Pipeline with a final RandomForestRegressor, compute per-tree predictions.
    Returns (mean_preds, std_preds). Otherwise returns (preds, zeros).
    """
    try:
        if hasattr(pipeline, "named_steps"):
            # separate preprocessing and final estimator
            steps = pipeline.named_steps
            preproc_items = list(steps.items())[:-1]
            from sklearn.pipeline import Pipeline
            if preproc_items:
                preproc = Pipeline(preproc_items)
                X_pre = preproc.transform(X_df)
            else:
                X_pre = X_df.values
            final = list(steps.values())[-1]
        else:
            X_pre = X_df.values
            final = pipeline

        if hasattr(final, "estimators_"):
            # RandomForestRegressor
            tree_preds = np.vstack([t.predict(X_pre) for t in final.estimators_])
            mean = tree_preds.mean(axis=0)
            std = tree_preds.std(axis=0)
            return mean, std
        else:
            preds = pipeline.predict(X_df)
            return np.array(preds), np.zeros_like(preds, dtype=float)
    except Exception:
        # fallback
        try:
            preds = pipeline.predict(X_df)
            return np.array(preds), np.zeros_like(preds, dtype=float)
        except Exception:
            return np.array([np.nan]), np.array([0.0])

# --------- Main action ---------
if submitted:
    # Basic input validation
    if distance_km < 0:
        st.error("Distance must be >= 0.")
    elif agent_age < 18:
        st.error("Agent age must be >= 18.")
    elif not (1.0 <= agent_rating <= 5.0):
        st.error("Agent rating must be between 1.0 and 5.0.")
    elif model is None:
        st.error("Model is not available. Fix MODEL_PATH or load the model.")
    else:
        # compute features
        feat_df = compute_features(distance_km, order_dt, agent_age, agent_rating, category, pickup_dt_val)
        st.subheader("Input features (used by model)")
        st.table(feat_df.T.rename(columns={0: "value"}))

        # predict
        preds_mean, preds_std = rf_uncertainty_if_rf(model, feat_df)
        pred = float(preds_mean[0])
        pred_std = float(preds_std[0]) if preds_std is not None else 0.0

        if np.isnan(pred):
            st.error("Prediction failed (see terminal for errors).")
        else:
            st.success(f"Estimated delivery time: **{pred:.2f} hours**")
            if pred_std > 0:
                st.info(f"Estimated uncertainty (±1 std): ±{pred_std:.2f} hours")

            # show top drivers if available
            if os.path.exists(FI_CSV):
                try:
                    fi = pd.read_csv(FI_CSV)
                    if "perm_importance_mean" in fi.columns and "feature" in fi.columns:
                        top3 = fi.sort_values("perm_importance_mean", ascending=False).head(3)
                        st.subheader("Top drivers from trained model (permutation importance)")
                        for _, r in top3.iterrows():
                            st.write(f"- **{r['feature']}** (importance: {r['perm_importance_mean']:.4f})")
                    else:
                        st.write("Feature importance file found but missing expected columns.")
                except Exception:
                    st.write("Feature importance file exists but couldn't be read.")
            else:
                st.info("Feature importance not available. Run STEP 9 to generate it.")

            st.markdown("**Operational tips**")
            st.write("- If predicted time is high, consider priority routing, notifying the customer, or assigning high-rated agents.")
            st.write("- Consider adjusting promised delivery windows by product category and order time.")

# Footer
st.markdown("---")
st.caption("Prototype app — for production: add authentication, input validation, versioning (MLflow), monitoring, and A/B testing.")
