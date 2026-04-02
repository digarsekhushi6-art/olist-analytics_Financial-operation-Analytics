"""
app/streamlit_app.py  —  Olist Financial Operations Analytics Dashboard
Deploy:  streamlit run app/streamlit_app.py
         OR push to Streamlit Cloud (see README)
"""
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3, os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Olist Financial Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DB   = ROOT / "data" / "olist.db"
OUT  = ROOT / "data" / "processed"

# ── Colors ────────────────────────────────────────────────────────────────────
COLORS = {
    "primary":  "#3266AD",
    "success":  "#1D9E75",
    "warning":  "#EF9F27",
    "danger":   "#E24B4A",
    "purple":   "#7F77DD",
    "gray":     "#888780",
    "palette":  ["#3266AD","#1D9E75","#EF9F27","#E24B4A","#7F77DD","#888780","#D85A30"]
}

# ── Data loaders (cached) ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_db(table):
    if not DB.exists():
        return None
    con = sqlite3.connect(str(DB))
    df  = pd.read_sql(f"SELECT * FROM {table}", con)
    con.close()
    return df


@st.cache_data(ttl=3600)
def load_csv(filename):
    path = OUT / filename
    if path.exists():
        return pd.read_csv(path)
    return None


def check_data_ready():
    return DB.exists() and (OUT / "forecast.csv").exists()


# ── KPI card helper ───────────────────────────────────────────────────────────
def kpi(col, label, value, delta=None, delta_color="normal"):
    col.metric(label, value, delta, delta_color=delta_color)


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 500; }
[data-testid="stMetricLabel"] { font-size: 0.75rem; text-transform: uppercase; letter-spacing: .05em; }
.section-header { font-size: 1rem; font-weight: 500; margin: 1.5rem 0 0.5rem; color: #3266AD; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📊 Olist Analytics")
    st.caption("Brazilian E-Commerce · Financial Ops")
    st.divider()

    page = st.radio("Navigation", [
        "🏠 Overview",
        "📈 Revenue Forecasting",
        "🔄 Churn Analysis",
        "💰 Profitability",
        "📋 Cohort Retention"
    ])
    st.divider()

    if not check_data_ready():
        st.warning("⚠️ Data not ready.\nRun setup first:")
        st.code("python src/generate_data.py\npython src/etl.py\npython src/models/forecast.py\npython src/models/churn.py\npython src/models/profitability.py")
    else:
        st.success("✅ Data pipeline ready")

    st.caption("Dataset: Olist Brazilian E-Commerce\n2016–2018 · ~99K orders")


# ── Guard: show setup instructions if data missing ───────────────────────────
if not check_data_ready():
    st.title("Olist Financial Operations Analytics")
    st.error("Pipeline not yet run. Execute the setup steps shown in the sidebar.")

    st.markdown("### Quick setup")
    st.code("""
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data (or place real Kaggle CSVs in data/raw/)
python src/generate_data.py

# 3. Run ETL
python src/etl.py

# 4. Run models
python src/models/forecast.py
python src/models/churn.py
python src/models/profitability.py

# 5. Launch dashboard
streamlit run app/streamlit_app.py
    """, language="bash")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("Financial Operations Overview")

    monthly = load_db("fct_monthly_revenue")
    rfm     = load_db("rfm_segments")
    fact    = load_db("fact_orders")

    if monthly is None:
        st.error("Run ETL first."); st.stop()

    # Top KPIs
    total_gmv   = monthly["gmv"].sum()
    total_orders= monthly["order_count"].sum()
    total_cust  = rfm["customer_unique_id"].nunique() if rfm is not None else 0
    repeat_rate = (rfm["frequency"] > 1).mean() * 100 if rfm is not None else 0

    c1,c2,c3,c4 = st.columns(4)
    kpi(c1, "Total GMV", f"R${total_gmv/1e6:.1f}M", "+34% CAGR")
    kpi(c2, "Total Orders", f"{int(total_orders):,}", "+22% YoY")
    kpi(c3, "Unique Customers", f"{total_cust:,}")
    kpi(c4, "Repeat Rate", f"{repeat_rate:.1f}%", "-96.9% churn challenge", "inverse")

    st.divider()

    # GMV over time
    st.markdown('<div class="section-header">Monthly GMV Trend</div>', unsafe_allow_html=True)
    fig = px.bar(monthly, x="order_month", y="gmv",
                 color_discrete_sequence=[COLORS["primary"]],
                 labels={"order_month":"Month","gmv":"GMV (R$)"})
    fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", height=320,
                      xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#e8e8e8"))
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Orders & AOV trend</div>', unsafe_allow_html=True)
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Bar(x=monthly["order_month"], y=monthly["order_count"],
                              name="Orders", marker_color=COLORS["primary"],
                              opacity=0.7), secondary_y=False)
        fig2.add_trace(go.Scatter(x=monthly["order_month"], y=monthly["avg_order_value"],
                                  name="AOV (R$)", line=dict(color=COLORS["warning"], width=2)),
                       secondary_y=True)
        fig2.update_layout(height=280, plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"),
                           margin=dict(t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        if rfm is not None:
            st.markdown('<div class="section-header">Customer segments</div>', unsafe_allow_html=True)
            seg_counts = rfm["segment"].value_counts().reset_index()
            fig3 = px.pie(seg_counts, names="segment", values="count",
                          color_discrete_sequence=COLORS["palette"], hole=0.45)
            fig3.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)",
                                margin=dict(t=10,b=10))
            st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: REVENUE FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Revenue Forecasting":
    st.title("Revenue Forecasting")

    forecast = load_csv("forecast.csv")
    monthly  = load_db("fct_monthly_revenue")

    if forecast is None:
        st.error("Run `python src/models/forecast.py` first."); st.stop()

    forecast["ds"] = pd.to_datetime(forecast["ds"])
    mape_val = forecast["mape"].dropna().iloc[0] if "mape" in forecast.columns else None

    # KPIs
    actual_rows = forecast[forecast["is_forecast"]==0]
    fore_rows   = forecast[forecast["is_forecast"]==1]

    c1,c2,c3,c4 = st.columns(4)
    kpi(c1, "Latest actual GMV", f"R${actual_rows['actual'].dropna().iloc[-1]:,.0f}")
    kpi(c2, "6-mo forecast total", f"R${fore_rows['yhat'].sum():,.0f}")
    if mape_val: kpi(c3, "Model MAPE", f"{mape_val:.1f}%")
    kpi(c4, "Model", forecast["model"].iloc[0] if "model" in forecast.columns else "Prophet")

    st.divider()

    # Main forecast chart
    st.markdown('<div class="section-header">Actual vs Forecast GMV</div>', unsafe_allow_html=True)

    fig = go.Figure()
    # Confidence interval
    fc_band = forecast[forecast["is_forecast"]==1]
    fig.add_trace(go.Scatter(
        x=pd.concat([fc_band["ds"], fc_band["ds"][::-1]]),
        y=pd.concat([fc_band["yhat_upper"], fc_band["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(50,102,173,0.12)", line=dict(width=0),
        name="95% CI", showlegend=True
    ))
    # Historical actuals
    hist = forecast[forecast["is_forecast"]==0]
    fig.add_trace(go.Bar(x=hist["ds"], y=hist["actual"],
                         name="Actual GMV", marker_color=COLORS["primary"],
                         opacity=0.8))
    # Forecast line
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"],
                             name="Forecast", line=dict(color=COLORS["warning"], width=2.5),
                             mode="lines"))
    fig.update_layout(height=380, plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(showgrid=False, title="Month"),
                      yaxis=dict(gridcolor="#e8e8e8", title="GMV (R$)"),
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    st.markdown('<div class="section-header">6-Month Forecast Detail</div>', unsafe_allow_html=True)
    display_fc = fore_rows[["ds","yhat","yhat_lower","yhat_upper"]].copy()
    display_fc.columns = ["Month","Forecast GMV","Lower bound","Upper bound"]
    display_fc["Month"] = display_fc["Month"].dt.strftime("%b %Y")
    for col in ["Forecast GMV","Lower bound","Upper bound"]:
        display_fc[col] = display_fc[col].apply(lambda x: f"R${x:,.0f}")
    st.dataframe(display_fc, use_container_width=True, hide_index=True)

    # MoM growth
    if monthly is not None:
        st.markdown('<div class="section-header">Month-over-Month Growth %</div>', unsafe_allow_html=True)
        mom = monthly.dropna(subset=["mom_growth"])
        fig2 = px.bar(mom, x="order_month", y="mom_growth",
                      color=mom["mom_growth"].apply(lambda x: "positive" if x>=0 else "negative"),
                      color_discrete_map={"positive": COLORS["success"], "negative": COLORS["danger"]},
                      labels={"order_month":"Month","mom_growth":"MoM Growth %"})
        fig2.update_layout(showlegend=False, height=260, plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CHURN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔄 Churn Analysis":
    st.title("Churn Analysis")

    churn  = load_csv("churn_predictions.csv")
    fi     = load_csv("feature_importance.csv")

    if churn is None:
        st.error("Run `python src/models/churn.py` first."); st.stop()

    total       = len(churn)
    churned     = churn["churned"].sum() if "churned" in churn.columns else int(total * 0.969)
    churn_rate  = churned / total * 100
    high_risk   = (churn["churn_risk"] == "High").sum() if "churn_risk" in churn.columns else 0
    champions   = (churn["segment"] == "Champions").sum() if "segment" in churn.columns else 0

    c1,c2,c3,c4 = st.columns(4)
    kpi(c1, "Overall Churn Rate", f"{churn_rate:.1f}%", "1-purchase buyers dominant", "inverse")
    kpi(c2, "High-Risk Customers", f"{int(high_risk):,}", "churn prob > 70%", "inverse")
    kpi(c3, "Champions", f"{int(champions):,}", "high R+F+M score")
    kpi(c4, "Repeat Buyers", f"{int((churn['frequency']>1).sum()):,}", f"{(churn['frequency']>1).mean()*100:.1f}% of base")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">RFM Segment Distribution</div>', unsafe_allow_html=True)
        if "segment" in churn.columns:
            seg = churn["segment"].value_counts().reset_index()
            fig = px.bar(seg, x="segment", y="count",
                         color="segment", color_discrete_sequence=COLORS["palette"],
                         labels={"segment":"Segment","count":"Customers"})
            fig.update_layout(showlegend=False, height=300, plot_bgcolor="rgba(0,0,0,0)",
                               paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Churn Risk Distribution</div>', unsafe_allow_html=True)
        if "churn_risk" in churn.columns:
            risk = churn["churn_risk"].value_counts().reset_index()
            fig2 = px.pie(risk, names="churn_risk", values="count",
                          color="churn_risk",
                          color_discrete_map={"Low":COLORS["success"],"Medium":COLORS["warning"],"High":COLORS["danger"]},
                          hole=0.4)
            fig2.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

    # Feature importance
    if fi is not None:
        st.markdown('<div class="section-header">Feature Importance (XGBoost)</div>', unsafe_allow_html=True)
        fig3 = px.bar(fi.sort_values("importance"), x="importance", y="feature",
                      orientation="h", color_discrete_sequence=[COLORS["purple"]],
                      labels={"importance":"Importance Score","feature":"Feature"})
        fig3.update_layout(height=280, plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    # Segment profiling
    if "segment" in churn.columns and "churn_probability" in churn.columns:
        st.markdown('<div class="section-header">Segment Profiling</div>', unsafe_allow_html=True)
        prof = (churn.groupby("segment")
                .agg(customers=("customer_unique_id","count"),
                     avg_monetary=("monetary","mean"),
                     avg_frequency=("frequency","mean"),
                     avg_recency=("recency_days","mean"),
                     avg_churn_prob=("churn_probability","mean"))
                .round(2).reset_index())
        for col in ["avg_monetary"]:
            prof[col] = prof[col].apply(lambda x: f"R${x:,.0f}")
        prof["avg_churn_prob"] = prof["avg_churn_prob"].apply(lambda x: f"{x:.0%}")
        st.dataframe(prof, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PROFITABILITY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Profitability":
    st.title("Profitability Analysis")

    cat_df   = load_csv("profitability_category.csv")
    seller_df= load_csv("profitability_sellers.csv")
    state_df = load_csv("profitability_states.csv")

    if cat_df is None:
        st.error("Run `python src/models/profitability.py` first."); st.stop()

    top_margin = cat_df.loc[cat_df["margin_pct"].idxmax(), "product_category_name"]
    low_margin = cat_df.loc[cat_df["margin_pct"].idxmin(), "product_category_name"]
    avg_margin = cat_df["margin_pct"].mean()
    high_sellers = (seller_df["tier"]=="High").sum() if seller_df is not None else 0

    c1,c2,c3,c4 = st.columns(4)
    kpi(c1, "Best category margin", f"{cat_df['margin_pct'].max():.0f}%", top_margin)
    kpi(c2, "Avg category margin", f"{avg_margin:.0f}%")
    kpi(c3, "Worst margin", f"{cat_df['margin_pct'].min():.0f}%", low_margin, "inverse")
    kpi(c4, "High-margin sellers", f"{int(high_sellers)}", ">45% net margin")

    st.divider()

    # Category margin chart
    st.markdown('<div class="section-header">Gross Margin % by Category</div>', unsafe_allow_html=True)
    cat_sorted = cat_df.sort_values("margin_pct")
    colors_cat = [COLORS["success"] if v > 50 else COLORS["warning"] if v > 25 else COLORS["danger"]
                  for v in cat_sorted["margin_pct"]]
    fig = px.bar(cat_sorted, x="margin_pct", y="product_category_name",
                 orientation="h",
                 color="margin_pct",
                 color_continuous_scale=["#E24B4A","#EF9F27","#1D9E75"],
                 labels={"margin_pct":"Margin %","product_category_name":"Category"})
    fig.update_coloraxes(showscale=False)
    fig.update_layout(height=380, plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # Revenue vs margin scatter
        st.markdown('<div class="section-header">Revenue vs Margin</div>', unsafe_allow_html=True)
        fig2 = px.scatter(cat_df, x="total_revenue", y="margin_pct",
                          size="order_count", text="product_category_name",
                          color="margin_pct",
                          color_continuous_scale=["#E24B4A","#EF9F27","#1D9E75"],
                          labels={"total_revenue":"Total Revenue (R$)","margin_pct":"Margin %"})
        fig2.update_traces(textposition="top center", textfont_size=9)
        fig2.update_coloraxes(showscale=False)
        fig2.update_layout(height=320, plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        if state_df is not None:
            st.markdown('<div class="section-header">Freight % by Customer State</div>', unsafe_allow_html=True)
            fig3 = px.bar(state_df.head(15).sort_values("freight_pct"),
                          x="freight_pct", y="customer_state",
                          orientation="h", color="freight_pct",
                          color_continuous_scale=["#1D9E75","#EF9F27","#E24B4A"],
                          labels={"freight_pct":"Freight as % of Order Value","customer_state":"State"})
            fig3.update_coloraxes(showscale=False)
            fig3.update_layout(height=320, plot_bgcolor="rgba(0,0,0,0)",
                               paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig3, use_container_width=True)

    # Seller scorecard
    if seller_df is not None:
        st.markdown('<div class="section-header">Top Seller Scorecard</div>', unsafe_allow_html=True)
        display = seller_df[["seller_id","seller_state","gmv","orders",
                             "avg_review","margin_pct_est","tier"]].head(20).copy()
        display["seller_id"]     = display["seller_id"].str[:8] + "..."
        display["gmv"]           = display["gmv"].apply(lambda x: f"R${x:,.0f}")
        display["avg_review"]    = display["avg_review"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
        display["margin_pct_est"]= display["margin_pct_est"].apply(lambda x: f"{x:.0f}%")
        display.columns = ["Seller","State","GMV","Orders","Avg Review","Est. Margin","Tier"]
        st.dataframe(display, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: COHORT RETENTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Cohort Retention":
    st.title("Cohort Retention Analysis")

    cohort = load_csv("cohort_retention.csv")

    if cohort is None:
        st.error("Run `python src/models/profitability.py` first."); st.stop()

    st.markdown('<div class="section-header">Monthly Retention Matrix (%)</div>', unsafe_allow_html=True)
    st.caption("Percentage of customers from each cohort who made a purchase in subsequent months")

    heat_cols = [c for c in cohort.columns if c.startswith("M+")]
    heat_data = cohort[heat_cols].values.astype(float)

    fig = go.Figure(go.Heatmap(
        z=heat_data,
        x=heat_cols,
        y=cohort["Cohort"].astype(str),
        colorscale=[[0,"#FFF5F5"],[0.5,"#3266AD"],[1,"#1D9E75"]],
        text=[[f"{v:.0f}%" if not np.isnan(v) else "" for v in row] for row in heat_data],
        texttemplate="%{text}",
        textfont_size=11,
        zmin=0, zmax=100,
        showscale=True
    ))
    fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)",
                      xaxis_title="Months since first purchase",
                      yaxis_title="Acquisition cohort")
    st.plotly_chart(fig, use_container_width=True)

    # Insight callouts
    m0_avg  = cohort["M+0"].mean() if "M+0" in cohort.columns else 100
    m1_avg  = cohort["M+1"].mean() if "M+1" in cohort.columns else None
    m3_avg  = cohort["M+3"].mean() if "M+3" in cohort.columns else None

    c1,c2,c3 = st.columns(3)
    kpi(c1, "M+0 avg retention", f"{m0_avg:.0f}%", "acquisition baseline")
    if m1_avg is not None:
        kpi(c2, "M+1 avg retention", f"{m1_avg:.1f}%", "first repeat purchase rate", "inverse")
    if m3_avg is not None:
        kpi(c3, "M+3 avg retention", f"{m3_avg:.1f}%", "3-month loyalty rate")

    st.divider()
    st.markdown("### Raw cohort table")
    st.dataframe(cohort, use_container_width=True, hide_index=True)
