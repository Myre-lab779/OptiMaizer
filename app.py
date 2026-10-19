import base64
import os
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st

st.set_page_config(
    page_title="Optimaizer | Corn Yield Calculator", page_icon="🌽", layout="wide",
    menu_items={"Get help": None, "Report a bug": None,
                "About": "Optimaizer: a regression-based corn yield prototype for the City of Ilagan. "
                         "The model is fitted to a simulated dataset."},
)

# ==============================================================================
# DEPLOYMENT SETTINGS (security)
# ==============================================================================
ENABLE_UPLOAD = False
MAX_UPLOAD_MB = 5
MAX_UPLOAD_ROWS = 5000
MAX_UPLOAD_COLS = 50
ALLOW_RAINFALL_OVERRIDE = False

# ==============================================================================
# MODEL DEFINITION
# ==============================================================================
DEFAULT_PARAMS = {
    "B0": 3835.25491030, "B1": 619.07847795, "B2": 5.95768991, "B3": -0.01864847,
    "B4": 1.14190054, "B5": 265.14618140, "F_mean": 317.00000000,
}
DEFAULT_RANGES = {"fert_range": (100.0, 600.0), "rain_range": (30.0, 1834.0)}
SEASON_RAIN = {0: 110.0, 1: 1155.0}

TERM_NAMES = {
    "B0": "Intercept",
    "B1": "Irrigation (Yes = 1)",
    "B2": "Fertilizer, centered (kg)",
    "B3": "Fertilizer, centered squared",
    "B4": "Rainfall (mm)",
    "B5": "Season (Wet = 1)",
    "F_mean": "Mean fertilizer used for centering (kg)",
}


# ==============================================================================
# HELPERS
# ==============================================================================
@st.cache_data
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None


class DataError(Exception):
    """A problem with an uploaded file that is safe to show to the user."""


def _clean(x, n=30):
    return re.sub(r"[^A-Za-z0-9 _.\-/()%]", "", str(x))[:n]


def find_col(columns, include, exclude=("cent", "sq", "interact")):
    for c in columns:
        name = str(c).lower()
        if any(k in name for k in include) and not any(k in name for k in exclude):
            return c
    return None


def to_binary_irrigation(series):
    yes = {"yes", "y", "true", "1", "1.0", "oo", "t"}
    no = {"no", "n", "false", "0", "0.0", "hindi", "f"}
    s = series.astype(str).str.strip().str.lower()
    values = s.isin(yes).astype(float)
    values = values.where(series.notna())
    unrecognized = sorted(set(s[series.notna() & ~s.isin(yes | no)]))
    return values, unrecognized


def to_binary_season(series):
    s = series.astype(str).str.strip().str.lower()
    wet = s.str.contains("wet") | s.isin(["w", "1", "1.0"])
    dry = s.str.contains("dry") | s.isin(["d", "0", "0.0"])
    values = wet.astype(float).where(series.notna())
    unrecognized = sorted(set(s[series.notna() & ~(wet | dry)]))
    return values, unrecognized


def train_model(df, yield_unit_choice, fert_unit_choice, kg_per_bag):
    cols = list(df.columns)
    found = {
        "Yield": find_col(cols, ["yield", "prod"]),
        "Irrigation": find_col(cols, ["irr"]),
        "Fertilizer": find_col(cols, ["fert"], exclude=("cent", "sq", "interact", "bag"))
                      or find_col(cols, ["fert"]),
        "Rainfall": find_col(cols, ["rain", "precip"]),
        "Season": find_col(cols, ["season"]),
    }
    missing = [k for k, v in found.items() if v is None]
    if missing:
        raise DataError("Could not find column(s) for: " + ", ".join(missing))

    notes = []
    irr, bad_irr = to_binary_irrigation(df[found["Irrigation"]])
    season, bad_season = to_binary_season(df[found["Season"]])

    data = pd.DataFrame({
        "yield": pd.to_numeric(df[found["Yield"]], errors="coerce"),
        "fert": pd.to_numeric(df[found["Fertilizer"]], errors="coerce"),
        "rain": pd.to_numeric(df[found["Rainfall"]], errors="coerce"),
        "irr": irr,
        "season": season,
    })
    data = data.dropna()
    if len(data) < 10:
        raise DataError("Not enough usable rows to train a model (need at least 10).")

    if yield_unit_choice == "Auto-detect":
        name = str(found["Yield"]).lower()
        yield_is_mt = ("mt" in name or "ton" in name) and "kg" not in name or data["yield"].median() < 100
    else:
        yield_is_mt = yield_unit_choice == "MT/ha"
    if yield_is_mt:
        data["yield"] = data["yield"] * 1000

    if fert_unit_choice == "Auto-detect":
        fert_is_bags = "bag" in str(found["Fertilizer"]).lower()
    else:
        fert_is_bags = fert_unit_choice == "Bags"
    if fert_is_bags:
        data["fert"] = data["fert"] * kg_per_bag

    f_mean = data["fert"].mean()
    X = pd.DataFrame({
        "Irrigation": data["irr"],
        "F_cent": data["fert"] - f_mean,
        "F_cent_sq": (data["fert"] - f_mean) ** 2,
        "Rainfall": data["rain"],
        "Season": data["season"],
    })
    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(data["yield"], X).fit()
    p = model.params
    params = {
        "B0": float(p["const"]), "B1": float(p["Irrigation"]),
        "B2": float(p["F_cent"]), "B3": float(p["F_cent_sq"]),
        "B4": float(p["Rainfall"]), "B5": float(p["Season"]),
        "F_mean": float(f_mean),
    }
    table = pd.DataFrame({
        "Term": ["Intercept", "Irrigation", "Fertilizer (centered)", "Fertilizer (centered²)", "Rainfall", "Season"],
        "Coefficient": model.params.values,
        "p-value": model.pvalues.values,
    })
    info = {
        "n": int(model.nobs),
        "r2": float(model.rsquared),
        "r2_adj": float(model.rsquared_adj),
        "table": table,
        "notes": notes,
        "fert_range": (float(data["fert"].min()), float(data["fert"].max())),
        "rain_range": (float(data["rain"].min()), float(data["rain"].max())),
    }
    return params, info


# ==============================================================================
# SESSION STATE
# ==============================================================================
if "model_params" not in st.session_state:
    st.session_state.model_params = dict(DEFAULT_PARAMS)
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "model_info" not in st.session_state:
    st.session_state.model_info = None
if "calculated_yield" not in st.session_state:
    st.session_state.calculated_yield = None
if "total_production" not in st.session_state:
    st.session_state.total_production = None
if "yield_solved" not in st.session_state:
    st.session_state.yield_solved = False
if "tp_solved" not in st.session_state:
    st.session_state.tp_solved = False

bg_img_1 = get_base64_image("image_f034c4.png")
map_bg = get_base64_image("map_bg.png")

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Settings")
    dark_mode = st.toggle("🌙 Dark Mode", value=False)
    st.divider()
    st.subheader("Units")
    kg_per_bag = st.number_input("Kilograms per fertilizer bag", min_value=1.0, max_value=200.0, value=50.0, step=1.0)

    with st.expander("Model coefficients"):
        mp_view = st.session_state.model_params
        coef
