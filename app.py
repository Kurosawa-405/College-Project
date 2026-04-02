# Run with: streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import shap
import ollama



# 🔥 CRITICAL FIX: NumPy compatibility patch
# Fixes errors from shap / prophet using deprecated numpy types
if not hasattr(np, "float"):
    np.float = np.float64
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool


# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="AI Employability Forecast",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Global CSS
# ---------------------------
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem; }

[data-testid="metric-container"] {
    background: #F54927;
    border: 1px solid #e8eaed;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
[data-testid="metric-container"] label {
    font-size: 0.75rem !important;
    color: #6b7280 !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    font-size: 1.6rem !important;
    font-weight: 600;
}

[data-testid="stSidebar"] {
    background: #f0f2f5;
    border-right: 1px solid #e2e4e9;
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSelectbox label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #111827;
    margin-bottom: 0.25rem;
}
.section-subtitle {
    font-size: 0.8rem;
    color: #6b7280;
    margin-bottom: 1rem;
}
.divider {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 1.5rem 0;
}
.topbar-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #111827;
}
.topbar-badge {
    background: #ecfdf5;
    color: #065f46;
    border: 1px solid #a7f3d0;
    border-radius: 999px;
    padding: 0.2rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-block;
}

/* Chat bubbles */
.chat-user {
    background: #2563eb;
    color: #fff;
    border-radius: 16px 16px 4px 16px;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0;
    max-width: 80%;
    margin-left: auto;
    font-size: 0.9rem;
}
.chat-assistant {
    background: #f3f4f6;
    color: #111827;
    border-radius: 16px 16px 16px 4px;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0;
    max-width: 85%;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Auth State
# ---------------------------
VALID_USERS = {
    "admin@example.com": "admin123",
    "user@example.com":  "user123",
}

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------
# LOGIN PAGE
# ---------------------------
def login_page():
    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        st.markdown("## 📊 AI Employability Forecast")
        st.markdown("##### Sign in to access your G20 dashboard")
        st.markdown("---")

        with st.form("login_form"):
            email    = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            st.markdown(" ")
            submitted = st.form_submit_button("Sign in →", use_container_width=True)

        if submitted:
            if email in VALID_USERS and VALID_USERS[email] == password:
                st.session_state.authenticated = True
                st.session_state.username = email
                st.rerun()
            else:
                st.error("Incorrect email or password. Please try again.")

        st.caption("Demo — **admin@example.com** / **admin123**")

# ---------------------------
# DATA & INDEX
# ---------------------------
@st.cache_data
def load_data():
    import os
    data_file = "G20_Historical_Data_2010_2024_v2.xlsx"
    if not os.path.exists(data_file):
        st.error(f"**Missing data file:** `{data_file}`\\n\\n"
                 "**Fix:** Ensure the Excel file is in the project root. "
                 "It contains G20 data 2010-2024. If lost, re-download from source.**")
        st.stop()
    
    df = pd.read_excel(data_file, header=2)

    df.columns = [
        "country", "Year", "gdp_per_capita", "hdi",
        "internet_penetration", "patents", "startups",
        "employment_rate", "automation_risk"
    ]
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    numeric_cols = [
        "gdp_per_capita", "hdi", "internet_penetration", "patents",
        "startups", "employment_rate", "automation_risk"
    ]
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(",", "")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    return df

def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

def compute_index(df):
    df = df.copy()
    df["gdp_n"]        = normalize(df["gdp_per_capita"])
    df["hdi_n"]        = normalize(df["hdi"])
    df["internet_n"]   = normalize(df["internet_penetration"])
    df["patents_n"]    = normalize(df["patents"])
    df["startups_n"]   = normalize(df["startups"])
    df["employment_n"] = normalize(df["employment_rate"])
    df["automation_n"] = normalize(df["automation_risk"])
    df["AI_employability"] = (
        0.20 * df["gdp_n"] +
        0.20 * df["hdi_n"] +
        0.15 * df["internet_n"] +
        0.15 * df["patents_n"] +
        0.10 * df["startups_n"] +
        0.10 * df["employment_n"] -
        0.10 * df["automation_n"]
    )
    return df

# ---------------------------
# CHART STYLE HELPER
# ---------------------------
def style_ax(ax, fig):
    fig.patch.set_facecolor("#f8f9fb")
    ax.set_facecolor("#f8f9fb")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35, color="#d1d5db")
    ax.tick_params(labelsize=9)

# ---------------------------
# OLLAMA HELPERS
# ---------------------------
@st.cache_data(ttl=30)
def get_ollama_models():
    """Return list of locally available Ollama model names."""
    try:
        result = ollama.list()
        # ollama.list() returns a dict with a 'models' key (list of dicts with 'name')
        models = result.get("models", [])
        return [m["name"] for m in models] if models else []
    except Exception:
        return []

def build_system_prompt(df: pd.DataFrame, global_df: pd.DataFrame) -> str:
    """Inject live dashboard statistics into the system prompt."""
    last_year = int(global_df["Year"].max())
    latest_global = global_df["AI_employability"].iloc[-1]
    earliest_global = global_df["AI_employability"].iloc[0]
    first_year = int(global_df["Year"].min())

    latest_year_df = df[df["Year"] == last_year].sort_values(
        "AI_employability", ascending=False
    ).reset_index(drop=True)

    top3 = latest_year_df[["country", "AI_employability"]].head(3).to_string(index=False)
    bot3 = latest_year_df[["country", "AI_employability"]].tail(3).to_string(index=False)

    trend = "upward" if latest_global > earliest_global else "downward"
    delta = latest_global - earliest_global

    return f"""You are an expert economic analyst assistant embedded in an AI Employability Forecast dashboard for G20 countries.

== Dashboard context ==
- Data range: {first_year}–{last_year}
- G20 average AI Employability Index: {earliest_global:.4f} ({first_year}) → {latest_global:.4f} ({last_year})  [{trend} trend, Δ={delta:+.4f}]
- Index formula: 20% GDP/capita + 20% HDI + 15% internet penetration + 15% patents + 10% startups + 10% employment rate − 10% automation risk (all min-max normalised)

Top 3 countries ({last_year}):
{top3}

Bottom 3 countries ({last_year}):
{bot3}

== Your role ==
Answer questions about the data, explain trends, compare countries, discuss implications for workforce policy, or interpret the index methodology. Be concise, data-grounded, and policy-aware. If asked about projections, note that the dashboard uses ARIMA and Prophet models.
"""

def stream_ollama(model: str, messages: list):
    """Yield text chunks from an Ollama streaming chat call."""
    stream = ollama.chat(model=model, messages=messages, stream=True)
    for chunk in stream:
        delta = chunk.get("message", {}).get("content", "")
        if delta:
            yield delta

# ---------------------------
# MAIN DASHBOARD
# ---------------------------
def dashboard():
    df        = load_data()
    df        = compute_index(df)
    global_df = df.groupby("Year")["AI_employability"].mean().reset_index()

    # ── Top bar ──
    c1, c2, c3 = st.columns([5, 2, 1])
    with c1:
        st.markdown('<div class="topbar-title">📊 AI Employability Forecast Dashboard</div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="topbar-badge">G20 · 2010–2024 · 10-Year Horizon</div>',
                    unsafe_allow_html=True)
    with c3:
        if st.button("Sign out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.chat_history = []
            st.rerun()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/combo-chart.png", width=48)
        st.markdown("### Navigation")
        page = st.radio(
            "Go to",
            ["📈 Overview", "🔮 Forecast", "🧠 SHAP Explainer", "💬 AI Analyst", "📄 Raw Data"],
            label_visibility="collapsed"
        )
        st.markdown("---")
        st.markdown("### Model settings")
        model_choice = st.selectbox("Forecast model", ["ARIMA", "Prophet"])
        scenario     = st.selectbox("Scenario", ["Base Case", "Optimistic", "Pessimistic"])
        show_all     = st.checkbox("Compare all scenarios", value=False)

        # ── Ollama model picker ──
        st.markdown("---")
        st.markdown("### Ollama (AI Analyst)")
        ollama_models = get_ollama_models()
        if ollama_models:
            ollama_model = st.selectbox("Local model", ollama_models, key="ollama_model")
        else:
            st.warning("No Ollama models found.\nRun `ollama pull llama3.2:3b` to get started.")
            ollama_model = None

        if st.button("Clear chat history", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        st.markdown("---")
        st.caption(f"Logged in as `{st.session_state.username}`")

    scenarios = {
        "Base Case":   {"arima_order": (2,1,2), "changepoint_prior": 0.05},
        "Optimistic":  {"arima_order": (2,1,1), "changepoint_prior": 0.50},
        "Pessimistic": {"arima_order": (2,2,2), "changepoint_prior": 0.001},
    }
    scenario_colors = {"Base Case": "orange", "Optimistic": "green", "Pessimistic": "red"}
    config = scenarios[scenario]
    color  = scenario_colors[scenario]

    forecast_steps = 12
    last_Year      = int(global_df["Year"].max())
    future_Years   = np.arange(last_Year + 1, last_Year + 1 + forecast_steps)
    all_Years      = np.arange(int(global_df["Year"].min()), last_Year + forecast_steps + 1)

    # ══════════════════════════════════════
    # PAGE: OVERVIEW
    # ══════════════════════════════════════
    if page == "📈 Overview":
        st.markdown('<div class="section-title">Global Overview</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">G20 average across all countries and years</div>',
                    unsafe_allow_html=True)

        latest_idx   = global_df["AI_employability"].iloc[-1]
        earliest_idx = global_df["AI_employability"].iloc[0]
        delta        = latest_idx - earliest_idx
        top_row      = df[df["Year"] == last_Year].sort_values("AI_employability", ascending=False)
        top_country  = top_row["country"].iloc[0] if len(top_row) else "—"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Index (Latest Year)", f"{latest_idx:.3f}", f"+{delta:.3f} vs {int(global_df['Year'].min())}")
        m2.metric("Countries tracked", "20")
        m3.metric("Years of data", f"{len(global_df)}")
        m4.metric("Top ranked country", top_country)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Historical AI Employability Index (G20 Average)</div>',
                    unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(11, 4))
        style_ax(ax, fig)
        ax.plot(global_df["Year"], global_df["AI_employability"],
                color="#2563eb", linewidth=2.5, marker="o", markersize=5, label="G20 Average")
        ax.fill_between(global_df["Year"], global_df["AI_employability"], alpha=0.1, color="#2563eb")
        ax.set_xticks(global_df["Year"].values)
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("Year", fontsize=10)
        ax.set_ylabel("AI Employability Index", fontsize=10)
        ax.legend(fontsize=9)
        st.pyplot(fig)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Country Rankings — Latest Year</div>',
                    unsafe_allow_html=True)

        latest = (
            df[df["Year"] == last_Year]
            .sort_values("AI_employability", ascending=False)
            [["country", "AI_employability", "gdp_per_capita", "hdi", "employment_rate"]]
            .reset_index(drop=True)
        )
        latest.index += 1
        latest.columns = ["Country", "AI Index", "GDP/Capita", "HDI", "Employment Rate"]
        st.dataframe(
            latest.style.format({
                "AI Index": "{:.4f}",
                "GDP/Capita": "{:,.0f}",
                "HDI": "{:.3f}",
                "Employment Rate": "{:.1f}"
            }).background_gradient(subset=["AI Index"], cmap="Blues"),
            use_container_width=True
        )

    # ══════════════════════════════════════
    # PAGE: FORECAST
    # ══════════════════════════════════════
    elif page == "🔮 Forecast":
        label = "All Scenarios" if show_all else scenario
        st.markdown('<div class="section-title">Forecast</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-subtitle">{model_choice} · {label} · {forecast_steps}-year horizon</div>',
                    unsafe_allow_html=True)

        if model_choice == "ARIMA":
            model_fit    = ARIMA(global_df["AI_employability"], order=config["arima_order"]).fit()
            forecast_sum = model_fit.get_forecast(steps=forecast_steps)
            forecast     = forecast_sum.predicted_mean
            conf_int     = forecast_sum.conf_int()

            fig, ax = plt.subplots(figsize=(11, 4))
            style_ax(ax, fig)
            ax.plot(global_df["Year"], global_df["AI_employability"],
                    label="Historical", color="#2563eb", linewidth=2.5)

            if show_all:
                for s_name, s_cfg in scenarios.items():
                    sc = scenario_colors[s_name]
                    m  = ARIMA(global_df["AI_employability"], order=s_cfg["arima_order"]).fit()
                    fc = m.get_forecast(steps=forecast_steps)
                    ci = fc.conf_int()
                    ax.plot(future_Years, fc.predicted_mean, linestyle="dashed",
                            label=s_name, color=sc, linewidth=2)
                    ax.fill_between(future_Years, ci.iloc[:,0], ci.iloc[:,1],
                                    alpha=0.1, color=sc)
            else:
                ax.plot(future_Years, forecast, linestyle="dashed",
                        label=f"Forecast ({scenario})", color=color, linewidth=2)
                ax.fill_between(future_Years, conf_int.iloc[:,0], conf_int.iloc[:,1],
                                alpha=0.18, color=color, label="Uncertainty interval")

            ax.axvline(x=last_Year, color="#9ca3af", linestyle=":", linewidth=1)
            ax.set_xticks(all_Years)
            ax.tick_params(axis="x", rotation=45)
            ax.set_xlabel("Year", fontsize=10)
            ax.set_ylabel("AI Employability Index", fontsize=10)
            ax.legend(fontsize=9)
            st.pyplot(fig)

            forecast_df = pd.DataFrame({
                "Year":        future_Years,
                "Forecast":    forecast.values.round(4),
                "Lower Bound": conf_int.iloc[:,0].values.round(4),
                "Upper Bound": conf_int.iloc[:,1].values.round(4),
            })

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Walk-Forward Validation (Last 3 Years)</div>',
                        unsafe_allow_html=True)

            train_size = len(global_df) - 3
            train      = global_df["AI_employability"][:train_size]
            test       = global_df["AI_employability"][train_size:]
            test_yrs   = global_df["Year"][train_size:].values
            arima_cv   = ARIMA(train, order=config["arima_order"]).fit()
            arima_pred = arima_cv.forecast(steps=3)

            mae  = mean_absolute_error(test, arima_pred)
            rmse = np.sqrt(mean_squared_error(test, arima_pred))
            mape = np.mean(np.abs((test.values - arima_pred.values) / test.values)) * 100

            c1, c2, c3 = st.columns(3)
            c1.metric("MAE",  f"{mae:.4f}")
            c2.metric("RMSE", f"{rmse:.4f}")
            c3.metric("MAPE", f"{mape:.2f}%")

            fig2, ax2 = plt.subplots(figsize=(10, 3.5))
            style_ax(ax2, fig2)
            ax2.plot(global_df["Year"], global_df["AI_employability"],
                     label="Historical", color="#2563eb", linewidth=2)
            ax2.plot(test_yrs, arima_pred.values, linestyle="dashed",
                     color="orange", linewidth=2, label="Backtested prediction")
            ax2.scatter(test_yrs, test.values, color="red", zorder=5, label="Actual")
            ax2.set_xticks(global_df["Year"].values)
            ax2.tick_params(axis="x", rotation=45)
            ax2.legend(fontsize=9)
            st.pyplot(fig2)

        elif model_choice == "Prophet":
            prophet_df       = global_df.rename(columns={"Year": "ds", "AI_employability": "y"})
            prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], format="%Y")
            model            = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=config["changepoint_prior"]
            )
            model.fit(prophet_df)
            future       = model.make_future_dataframe(periods=forecast_steps, freq="YE")
            forecast_raw = model.predict(future)
            future_mask  = forecast_raw["ds"].dt.year > last_Year

            fig, ax = plt.subplots(figsize=(11, 4))
            style_ax(ax, fig)
            ax.plot(global_df["Year"], global_df["AI_employability"],
                    label="Historical", color="#2563eb", linewidth=2.5)

            if show_all:
                for s_name, s_cfg in scenarios.items():
                    sc  = scenario_colors[s_name]
                    pm  = Prophet(
                        yearly_seasonality=False,
                        weekly_seasonality=False,
                        daily_seasonality=False,
                        changepoint_prior_scale=s_cfg["changepoint_prior"]
                    )
                    pm.fit(prophet_df.copy())
                    pf   = pm.make_future_dataframe(periods=forecast_steps, freq="YE")
                    pfc  = pm.predict(pf)
                    mask = pfc["ds"].dt.year > last_Year
                    ax.plot(pfc.loc[mask,"ds"].dt.year, pfc.loc[mask,"yhat"],
                            linestyle="dashed", label=s_name, color=sc, linewidth=2)
                    ax.fill_between(pfc.loc[mask,"ds"].dt.year,
                                    pfc.loc[mask,"yhat_lower"], pfc.loc[mask,"yhat_upper"],
                                    alpha=0.1, color=sc)
            else:
                ax.plot(forecast_raw.loc[future_mask,"ds"].dt.year,
                        forecast_raw.loc[future_mask,"yhat"],
                        linestyle="dashed", color=color, linewidth=2,
                        label=f"Forecast ({scenario})")
                ax.fill_between(forecast_raw.loc[future_mask,"ds"].dt.year,
                                forecast_raw.loc[future_mask,"yhat_lower"],
                                forecast_raw.loc[future_mask,"yhat_upper"],
                                alpha=0.18, color=color, label="Uncertainty interval")

            ax.axvline(x=last_Year, color="#9ca3af", linestyle=":", linewidth=1)
            ax.set_xticks(all_Years)
            ax.tick_params(axis="x", rotation=45)
            ax.set_xlabel("Year", fontsize=10)
            ax.set_ylabel("AI Employability Index", fontsize=10)
            ax.legend(fontsize=9)
            st.pyplot(fig)

            forecast_df = forecast_raw.loc[future_mask, ["ds","yhat","yhat_lower","yhat_upper"]].copy()
            forecast_df["Year"] = forecast_df["ds"].dt.year
            forecast_df = forecast_df.rename(columns={
                "yhat": "Forecast", "yhat_lower": "Lower Bound", "yhat_upper": "Upper Bound"
            })[["Year","Forecast","Lower Bound","Upper Bound"]]

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Prophet Cross Validation</div>',
                        unsafe_allow_html=True)
            with st.spinner("Running cross validation…"):
                df_cv = cross_validation(model, initial="2555 Days",
                                         period="365 Days", horizon="1095 Days")
                df_p  = performance_metrics(df_cv)
            st.dataframe(df_p[["horizon","mae","rmse","mape"]].round(4), use_container_width=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">Forecast Table — {label}</div>',
                    unsafe_allow_html=True)
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════
    # PAGE: SHAP EXPLAINER
    # ══════════════════════════════════════
    elif page == "🧠 SHAP Explainer":
        st.markdown('<div class="section-title">Explainable AI — Feature Importance (SHAP)</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Which features drive the AI Employability Index the most?</div>',
                    unsafe_allow_html=True)

        @st.cache_data
        def compute_shap(_df):
            features = ["gdp_n","hdi_n","internet_n","patents_n","startups_n","employment_n","automation_n"]
            X = _df[features]
            y = _df["AI_employability"]
            gbr = GradientBoostingRegressor(random_state=42).fit(X, y)
            explainer = shap.TreeExplainer(gbr)
            return explainer(X)

        shap_vals = compute_shap(df)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-title">Mean SHAP importance</div>',
                        unsafe_allow_html=True)
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            style_ax(ax1, fig1)
            shap.plots.bar(shap_vals, ax=ax1, show=False)
            st.pyplot(fig1)

        with col_b:
            st.markdown('<div class="section-title">Beeswarm — directional impact</div>',
                        unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            style_ax(ax2, fig2)
            shap.plots.beeswarm(shap_vals, show=False)
            st.pyplot(fig2)



    # ══════════════════════════════════════
    # PAGE: AI ANALYST (OLLAMA CHAT)
    # ══════════════════════════════════════
    elif page == "💬 AI Analyst":
        st.markdown('<div class="section-title">AI Analyst</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Ask questions about the G20 data, trends, and policy implications — '
            'powered by your local Ollama model</div>',
            unsafe_allow_html=True
        )

        if not ollama_model:
            st.warning("**Ollama not available.** Install from https://ollama.com/download and run `ollama pull llama3`. Chat disabled.")


        # ── Render chat history ──
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">{msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])

        # ── Suggested prompts (shown only when chat is empty) ──
        if not st.session_state.chat_history:
            st.markdown("**Suggested questions:**")
            suggestions = [
                "Which G20 country improved most between 2010 and 2024?",
                "What are the main drivers of the AI employability index?",
                "How does automation risk affect the index score?",
                "What policies could help lower-ranked countries improve?",
            ]
            cols = st.columns(2)
            for i, suggestion in enumerate(suggestions):
                if cols[i % 2].button(suggestion, key=f"suggest_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": suggestion})
                    st.rerun()

        # ── Chat input ──
        user_input = st.chat_input("Ask anything about the G20 AI Employability data…")

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Build message list: system prompt + full history
            system_prompt = build_system_prompt(df, global_df)
            messages_for_ollama = [{"role": "system", "content": system_prompt}] + \
                                   st.session_state.chat_history

            # Stream the response
            with st.chat_message("assistant"):
                response_text = st.write_stream(
                    stream_ollama(ollama_model, messages_for_ollama)
                )

            st.session_state.chat_history.append(
                {"role": "assistant", "content": response_text}
            )
            st.rerun()

    # ══════════════════════════════════════
    # PAGE: RAW DATA
    # ══════════════════════════════════════
    elif page == "📄 Raw Data":
        st.markdown('<div class="section-title">Raw Dataset</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Processed G20 data with computed AI Employability Index</div>',
                    unsafe_allow_html=True)

        countries = ["All"] + sorted(df["country"].unique().tolist())
        sel       = st.selectbox("Filter by country", countries)
        view_df   = df if sel == "All" else df[df["country"] == sel]

        st.dataframe(
            view_df.sort_values(["country","Year"])
                   .reset_index(drop=True)
                   .style.format({
                       "AI_employability": "{:.4f}",
                       "gdp_per_capita":   "{:,.0f}"
                   }),
            use_container_width=True
        )
        st.download_button(
            "⬇️ Download as CSV",
            data=view_df.to_csv(index=False),
            file_name="ai_employability_data.csv",
            mime="text/csv"
        )

# ---------------------------
# ROUTER
# ---------------------------
if not st.session_state.authenticated:
    login_page()
else:
    dashboard() 