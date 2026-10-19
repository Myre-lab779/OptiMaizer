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
# False = public-safe: visitors cannot upload files, so nobody can send data to the app.
# Set to True only when you run the app on your own computer.
ENABLE_UPLOAD = False
MAX_UPLOAD_MB = 5          # only used when ENABLE_UPLOAD = True
MAX_UPLOAD_ROWS = 5000
MAX_UPLOAD_COLS = 50
# False = rainfall is set by the system from the season (the public design: 3 user inputs).
# Set True ONLY on your own computer, to type each test farm's rainfall when validating the app.
ALLOW_RAINFALL_OVERRIDE = False

# ==============================================================================
# MODEL DEFINITION
#   Y = B0 + B1(Irrigation 0/1) + B2(Fc) + B3(Fc^2) + B4(Rainfall) + B5(Season)
#   Fc = fertilizer (kg) minus the mean fertilizer of the training data
#   Y  = yield in kg/ha
# ==============================================================================
# Defaults = finalized equation, fitted on the 240-farm TRAINING set of the SIMULATED dataset
# (generate_dataset.py, seed 2026; 80:20 split, random_state=42). These are the coefficients printed by
# corn_model_analysis.py (default_params.txt). If your printout differs, paste yours here.
DEFAULT_PARAMS = {
    "B0": 3835.25491030, "B1": 619.07847795, "B2": 5.95768991, "B3": -0.01864847,
    "B4": 1.14190054, "B5": 265.14618140, "F_mean": 317.00000000,
}
DEFAULT_RANGES = {"fert_range": (100.0, 600.0), "rain_range": (30.0, 1834.0)}   # range of the simulated data
SEASON_RAIN = {0: 110.0, 1: 1155.0}   # rainfall (mm) the system uses for each season = seasonal mean of the dataset

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
    """Strip anything unusual from text that came from an uploaded file before showing it."""
    return re.sub(r"[^A-Za-z0-9 _.\-/()%]", "", str(x))[:n]


def find_col(columns, include, exclude=("cent", "sq", "interact")):
    """First column whose name contains a keyword in `include`, ignoring derived columns."""
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
    """Fit the regression on an uploaded dataset. Returns (params, info). Raises DataError on bad data."""
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
        raise DataError(
            "Could not find column(s) for: " + ", ".join(missing)
            + ". Column names should contain: yield, irrigation, fertilizer, rainfall, season "
            "(use the raw fertilizer column, not the centered one)."
        )

    notes = []
    irr, bad_irr = to_binary_irrigation(df[found["Irrigation"]])
    season, bad_season = to_binary_season(df[found["Season"]])
    if bad_irr:
        notes.append(f"Irrigation values not recognized (read as 0): {[_clean(v) for v in bad_irr[:6]]}")
    if bad_season:
        notes.append(f"Season values not recognized (read as Dry): {[_clean(v) for v in bad_season[:6]]}")

    data = pd.DataFrame({
        "yield": pd.to_numeric(df[found["Yield"]], errors="coerce"),
        "fert": pd.to_numeric(df[found["Fertilizer"]], errors="coerce"),
        "rain": pd.to_numeric(df[found["Rainfall"]], errors="coerce"),
        "irr": irr,
        "season": season,
    })
    n_before = len(data)
    data = data.dropna()
    if n_before - len(data) > 0:
        notes.append(f"Dropped {n_before - len(data)} row(s) with missing or non-numeric values.")
    if len(data) < 10:
        raise DataError("Not enough usable rows to train a model (need at least 10).")

    # Yield unit -> kg/ha
    if yield_unit_choice == "Auto-detect":
        name = str(found["Yield"]).lower()
        named_mt = ("mt" in name or "ton" in name) and "kg" not in name
        yield_is_mt = named_mt or data["yield"].median() < 100
    else:
        yield_is_mt = yield_unit_choice == "MT/ha"
    if yield_is_mt:
        data["yield"] = data["yield"] * 1000
    notes.append(f"Yield read as {'MT/ha (converted to kg/ha)' if yield_is_mt else 'kg/ha'}.")

    # Fertilizer unit -> kg
    if fert_unit_choice == "Auto-detect":
        fert_is_bags = "bag" in str(found["Fertilizer"]).lower()
    else:
        fert_is_bags = fert_unit_choice == "Bags"
    if fert_is_bags:
        data["fert"] = data["fert"] * kg_per_bag
    notes.append(
        f"Fertilizer column '{_clean(found['Fertilizer'])}' read as "
        + (f"bags (x {kg_per_bag:g} kg)." if fert_is_bags else "kg.")
    )

    f_mean = data["fert"].mean()
    X = pd.DataFrame({
        "Irrigation": data["irr"],
        "F_cent": data["fert"] - f_mean,
        "F_cent_sq": (data["fert"] - f_mean) ** 2,
        "Rainfall": data["rain"],
        "Season": data["season"],
    })
    X = sm.add_constant(X, has_constant="add")
    if np.linalg.matrix_rank(X.values) < X.shape[1]:
        raise DataError(
            "The predictors are perfectly collinear or one has no variation "
            "(for example, every row has the same irrigation or season value)."
        )

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


def signed(value, digits):
    return f"- {abs(value):.{digits}f}" if value < 0 else f"+ {value:.{digits}f}"


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

    uploaded_file = None
    fert_unit_choice, yield_unit_choice = "Auto-detect", "Auto-detect"
    if ENABLE_UPLOAD:                                   # hidden in the public deployment
        fert_unit_choice = st.radio("Fertilizer column in your data is in", ["Auto-detect", "kg", "Bags"])
        yield_unit_choice = st.radio("Yield column in your data is in", ["Auto-detect", "kg/ha", "MT/ha"])
        st.divider()
        st.caption("⚠️ Do not upload private or personal data to a shared app.")
        uploaded_file = st.file_uploader("Upload Historical Data", type=["csv", "xlsx"])

        template = pd.DataFrame({
            "Irrigation": ["Yes", "No"],
            "Fertilizer_kg": [250, 150],
            "Rainfall_mm": [180, 45],
            "Season": ["Wet", "Dry"],
            "Yield_kg_ha": [7200, 5100],
        })
        st.download_button("⬇️ Sample CSV template", template.to_csv(index=False),
                           file_name="corn_data_template.csv", mime="text/csv")

    if uploaded_file is not None:
        try:
            if uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
                raise DataError(f"The file is larger than {MAX_UPLOAD_MB} MB.")
            if uploaded_file.name.lower().endswith(".csv"):
                raw_df = pd.read_csv(uploaded_file, nrows=MAX_UPLOAD_ROWS + 1)
            else:
                raw_df = pd.read_excel(uploaded_file, nrows=MAX_UPLOAD_ROWS + 1)
            if raw_df.shape[0] > MAX_UPLOAD_ROWS or raw_df.shape[1] > MAX_UPLOAD_COLS:
                raise DataError(f"The file is too large (limit: {MAX_UPLOAD_ROWS} rows and {MAX_UPLOAD_COLS} columns).")
            params, info = train_model(raw_df, yield_unit_choice, fert_unit_choice, kg_per_bag)
            st.session_state.model_params = params
            st.session_state.model_info = info
            st.session_state.model_trained = True
            st.success(f"✅ Model trained on {info['n']} rows.")
            for note in info["notes"]:
                st.text("• " + note)                    # plain text: file content is never rendered as markdown/HTML
        except DataError as e:
            st.session_state.model_trained = False
            st.error(f"⚠️ {e}")
        except Exception:                               # never show raw error details to visitors
            st.session_state.model_trained = False
            st.error("The file could not be processed. Check that it is a valid CSV or Excel file with the required columns.")

    with st.expander("Model coefficients"):
        mp_view = st.session_state.model_params
        coef_df = pd.DataFrame({
            "Term": [TERM_NAMES[k] for k in TERM_NAMES],
            "Value": [mp_view[k] for k in TERM_NAMES],
        })
        st.dataframe(coef_df, hide_index=True, width="stretch")
        info = st.session_state.model_info
        if st.session_state.model_trained and info:
            st.caption(f"n = {info['n']}  |  R² = {info['r2']:.4f}  |  Adj. R² = {info['r2_adj']:.4f}")
            st.dataframe(info["table"], hide_index=True, width="stretch")
        if st.button("Reset to default coefficients"):
            st.session_state.model_params = dict(DEFAULT_PARAMS)
            st.session_state.model_trained = False
            st.session_state.model_info = None
            st.session_state.calculated_yield = None
            st.session_state.total_production = None
            st.session_state.tp_solved = False
            st.rerun()

# ==============================================================================
# THEME
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
    .main label, .main p, .main span, .main div {{
        color: {text_color} !important;
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
if ENABLE_UPLOAD and not st.session_state.model_trained:
    st.info("💡 **Tip:** The default coefficients are the finalized model fitted on the 240-farm training set. "
            "Uploading a file in the sidebar retrains the model on all rows of that file.")

st.markdown("<div class='opt-title'>Optimaizer</div>", unsafe_allow_html=True)
st.markdown("<div class='opt-subtitle'>Corn Yield Calculator</div>", unsafe_allow_html=True)
st.markdown("<div class='opt-desc'>A REGRESSION-BASED PROTOTYPE DESIGNED TO ESTIMATE CORN YIELD IN THE CITY OF ILAGAN. IT USES IRRIGATION, FERTILIZER USE, AND CROPPING SEASON, WITH SEASONAL RAINFALL SET BY THE SYSTEM, TO PROVIDE MODEL-BASED YIELD ESTIMATES. THE MODEL IS FITTED TO A SIMULATED DATASET.</div>", unsafe_allow_html=True)

st.divider()

# ==============================================================================
# SECTION 1: YIELD
# ==============================================================================
st.markdown("<div class='section-header'>🌾 YIELD</div>", unsafe_allow_html=True)

y_col1, y_col2 = st.columns([1, 1], gap="large")

with y_col1:
    season_choice = st.selectbox("Cropping Season", ["Dry Season (0)", "Wet Season (1)"])
    s_val = 1 if "Wet" in season_choice else 0

    if ALLOW_RAINFALL_OVERRIDE:                         # testing only
        r_val = st.number_input("Rainfall (mm) - testing only", min_value=0.0, max_value=5000.0,
                                value=SEASON_RAIN[s_val], step=1.0)
    else:
        r_val = SEASON_RAIN[s_val]
        st.markdown(f"**Rainfall (set by the system for this season):** {r_val:,.0f} mm")
        st.caption("Seasonal mean rainfall of the dataset. You do not need to enter it.")

    irr_choice = st.selectbox("Irrigation", ["No (0)", "Yes (1)"])
    i_val = 1 if "Yes" in irr_choice else 0

    f_bags = st.number_input("Fertilizer (Number of Bags per Hectare)", min_value=0.0, max_value=100.0, value=4.0, step=0.5)

    farm_area = st.number_input("Farm Area (Hectares)", min_value=0.1, max_value=1000.0, value=1.0, step=0.1)

    st.markdown("---")
    st.markdown("**CONVERT TO METRIC TONS?**")
    yield_mt = st.checkbox("Show Yield in MT/ha", key="y_mt")
    st.caption("This is all in the assumption that the soil pH, soil moisture, and temperature is in typical growing conditions in the City of Ilagan.")

current_inputs = (i_val, f_bags, r_val, s_val, kg_per_bag, farm_area)

with y_col2:
    if st.button("SOLVE", width="stretch", type="secondary"):
        f_kg = f_bags * kg_per_bag
        f_cent = f_kg - mp["F_mean"]

        y_calc = (mp["B0"]
                  + mp["B1"] * i_val
                  + mp["B2"] * f_cent + mp["B3"] * (f_cent ** 2)
                  + mp["B4"] * r_val
                  + mp["B5"] * s_val)

        calc_yield = max(y_calc, 0.0)
        tot_prod = calc_yield * farm_area

        st.session_state.raw_yield = y_calc
        st
