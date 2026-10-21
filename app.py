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
        coef_df = pd.DataFrame({
            "Term": [TERM_NAMES[k] for k in TERM_NAMES],
            "Value": [mp_view[k] for k in TERM_NAMES],
        })
        st.dataframe(coef_df, hide_index=True, width="stretch")
        if st.button("Reset to default coefficients"):
            st.session_state.model_params = dict(DEFAULT_PARAMS)
            st.session_state.model_trained = False
            st.session_state.model_info = None
            st.session_state.calculated_yield = None
            st.session_state.total_production = None
            st.session_state.yield_solved = False
            st.session_state.tp_solved = False
            st.rerun()

# ==============================================================================
# THEME CONFIGURATION
# ==============================================================================
if dark_mode:
    bg_color = "#111812"
    text_color = "#E8F8F5"
    title_color = "#2ECC71"
    subtitle_color = "#58D68D"
    desc_color = "#A3E4D7"
    input_bg = "#1E2B22"
    input_border = "#27AE60"
    input_text = "#FFFFFF"
    result_bg = "#1B3B2B"
    result_border = "#27AE60"
    result_text = "#2ECC71"
    gradient_overlay = "rgba(17, 24, 18, 0.94)"
else:
    bg_color = "#F8F9FA"
    text_color = "#2C3E50"
    title_color = "#27AE60"
    subtitle_color = "#1E8449"
    desc_color = "#566573"
    input_bg = "#FFFFFF"
    input_border = "#D5D8DC"
    input_text = "#2C3E50"
    result_bg = "#EAFAF1"
    result_border = "#A9DFBF"
    result_text = "#1E8449"
    gradient_overlay = "rgba(248, 249, 250, 0.92)"

st.markdown(f"""
    <style>
    .stApp {{
        background-color: {bg_color} !important;
        color: {text_color} !important;
        background-image: linear-gradient({gradient_overlay}, {gradient_overlay}), url("data:image/png;base64,{map_bg if map_bg else ''}");
        background-size: 60% auto;
        background-position: center bottom;
        background-repeat: no-repeat;
    }}
    .opt-title {{
        font-family: 'Georgia', serif;
        font-size: 4.5rem;
        color: {title_color} !important;
        font-weight: bold;
        margin-bottom: -15px;
    }}
    .opt-subtitle {{
        font-family: 'Georgia', serif;
        font-size: 2.2rem;
        color: {subtitle_color} !important;
        margin-bottom: 15px;
    }}
    .opt-desc {{
        font-family: 'Arial', sans-serif;
        font-size: 0.85rem;
        color: {desc_color} !important;
        font-weight: bold;
        text-transform: uppercase;
        max-width: 850px;
        margin-bottom: 30px;
    }}
    .section-header {{
        font-family: 'Georgia', serif;
        font-size: 2.8rem;
        color: {subtitle_color} !important;
        margin-top: 10px;
        margin-bottom: 20px;
    }}
    .main label, .main p, .main span, .stMarkdown, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {{
        color: {text_color} !important;
    }}
    .stCaption {{
        color: {desc_color} !important;
    }}
    div[data-baseweb="input"] input, div[data-baseweb="select"] div {{
        color: {input_text} !important;
    }}
    div[data-baseweb="input"], div[data-baseweb="select"] {{
        background-color: {input_bg} !important;
        border: 1px solid {input_border} !important;
        border-radius: 6px;
    }}
    .stButton > button {{
        background-color: #27AE60 !important;
        color: #FFFFFF !important;
        border: 1px solid #27AE60 !important;
        font-weight: bold !important;
    }}
    .stButton > button:hover {{
        background-color: #219653 !important;
        color: #FFFFFF !important;
        border: 1px solid #219653 !important;
    }}
    .result-box {{
        background-color: {result_bg} !important;
        padding: 20px;
        border-radius: 6px;
        border: 1px solid {result_border};
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        color: {result_text} !important;
        margin-bottom: 15px;
    }}
    .top-banner {{
        width: 100%;
        height: 120px;
        background-image: url("data:image/png;base64,{bg_img_1 if bg_img_1 else ''}");
        background-size: cover;
        background-position: center;
        margin-bottom: 20px;
    }}
    </style>
""", unsafe_allow_html=True)

if bg_img_1:
    st.markdown("<div class='top-banner'></div>", unsafe_allow_html=True)

mp = st.session_state.model_params

st.warning("**Prototype.** The model was fitted to a **simulated dataset**, not real farm records. "
           "Results are model-based illustrations and should not be used as farm recommendations.")

st.markdown("<div class='opt-title'>Optimaizer</div>", unsafe_allow_html=True)
st.markdown("<div class='opt-subtitle'>Corn Yield Calculator</div>", unsafe_allow_html=True)
st.markdown("<div class='opt-desc'>A REGRESSION-BASED PROTOTYPE DESIGNED TO ESTIMATE CORN YIELD IN THE CITY OF ILAGAN. IT USES IRRIGATION, FERTILIZER USE, AND CROPPING SEASON, WITH SEASONAL RAINFALL SET BY THE SYSTEM, TO PROVIDE MODEL-BASED YIELD ESTIMATES. THE MODEL IS FITTED TO A SIMULATED DATASET.</div>", unsafe_allow_html=True)

st.divider()

# ==============================================================================
# SECTION 1: YIELD (Per Hectare) - Exactly 3 inputs
# ==============================================================================
st.markdown("<div class='section-header'>🌽 YIELD</div>", unsafe_allow_html=True)

y_col1, y_col2 = st.columns([1, 1], gap="large")

with y_col1:
    season_choice = st.selectbox("Cropping Season", ["Dry Season (0)", "Wet Season (1)"])
    s_val = 1 if "Wet" in season_choice else 0

    r_val = SEASON_RAIN[s_val]
    st.markdown(f"**Rainfall (set by the system for this season):** {r_val:,.0f} mm")
    st.caption("Seasonal mean rainfall of the dataset. You do not need to enter it.")

    irr_choice = st.selectbox("Irrigation", ["No (0)", "Yes (1)"])
    i_val = 1 if "Yes" in irr_choice else 0

    f_bags = st.number_input("Fertilizer (Number of Bags per Hectare)", min_value=0.0, max_value=100.0, value=4.0, step=0.5)

    st.markdown("---")
    st.markdown("**CONVERT TO METRIC TONS?**")
    yield_mt = st.checkbox("Show Yield in MT/ha", key="y_mt")
    st.caption("This is all in the assumption that the soil pH, soil moisture, and temperature is in typical growing conditions in the City of Ilagan.")

with y_col2:
    if st.button("SOLVE YIELD", width="stretch", type="secondary"):
        f_kg = f_bags * kg_per_bag
        f_cent = f_kg - mp["F_mean"]

        y_calc = (mp["B0"]
                  + mp["B1"] * i_val
                  + mp["B2"] * f_cent + mp["B3"] * (f_cent ** 2)
                  + mp["B4"] * r_val
                  + mp["B5"] * s_val)

        st.session_state.calculated_yield = max(y_calc, 0.0)
        st.session_state.yield_solved = True

    if st.session_state.get("yield_solved", False):
        y_val = st.session_state.calculated_yield
        y_disp = f"{y_val / 1000:,.2f} MT/ha" if yield_mt else f"{y_val:,.2f} kg/ha"
        st.markdown(f"<div class='result-box'>🌽 Calculated Yield: {y_disp}</div>", unsafe_allow_html=True)

st.divider()

# ==============================================================================
# SECTION 2: TOTAL PRODUCTION (Separate Area)
# ==============================================================================
st.markdown("<div class='section-header'>🚜 TOTAL PRODUCTION</div>", unsafe_allow_html=True)

tp_col1, tp_col2 = st.columns([1, 1], gap="large")

with tp_col1:
    farm_area = st.number_input("Farm Area (Hectares)", min_value=0.1, max_value=1000.0, value=1.0, step=0.1)
    st.caption("Enter your total farm land area to scale the per-hectare yield into total harvest production.")

    st.markdown("---")
    st.markdown("**CONVERT TO METRIC TONS?**")
    prod_mt = st.checkbox("Show Total Production in Metric Tons (MT)", key="tp_mt")

with tp_col2:
    if st.button("SOLVE TOTAL PRODUCTION", width="stretch", type="secondary"):
        if st.session_state.get("calculated_yield") is not None:
            st.session_state.total_production = st.session_state.calculated_yield * farm_area
            st.session_state.tp_solved = True
        else:
            st.warning("Please solve for the Yield first in Section 1 above.")

    if st.session_state.get("tp_solved", False) and st.session_state.get("total_production") is not None:
        tp_val = st.session_state.total_production
        tp_disp = f"{tp_val / 1000:,.2f} MT" if prod_mt else f"{tp_val:,.2f} kg"
        st.markdown(f"<div class='result-box'>🚜 Estimated Total Production: {tp_disp}</div>", unsafe_allow_html=True)
