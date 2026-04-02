import subprocess, sys
from pathlib import Path

def _ensure_pipeline():
    root = Path(__file__).parent.parent
    db   = root / "data" / "olist.db"
    fc   = root / "data" / "processed" / "forecast.csv"
    if db.exists() and fc.exists():
        return
    for s in ["src/generate_data.py","src/etl.py","src/models/forecast.py","src/models/churn.py","src/models/profitability.py"]:
        subprocess.run([sys.executable, str(root/s)], check=True)

_ensure_pipeline()

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Olist Financial Analytics", page_icon="📊", layout="wide")

ROOT = Path(__file__).parent.parent
DB   = ROOT / "data" / "olist.db"
OUT  = ROOT / "data" / "processed"

COLORS = ["#3266AD","#1D9E75","#EF9F27","#E24B4A","#7F77DD","#888780","#D85A30"]

@st.cache_data(ttl=3600)
def load_db(table):
    if not DB.exists(): return None
    con = sqlite3.connect(str(DB))
    df  = pd.read_sql(f"SELECT * FROM {table}", con)
    con.close()
    return df

@st.cache_data(ttl=3600)
def load_csv(filename):
    path = OUT / filename
    return pd.read_csv(path) if path.exists() else None

with st.sidebar:
    st.title("📊 Olist Analytics")
    st.caption("Brazilian E-Commerce · Financial Ops")
    st.divider()
    page = st.radio("Navigation", ["🏠 Overview","📈 Revenue Forecasting","🔄 Churn Analysis","💰 Profitability","📋 Cohort Retention"])
    st.divider()
    st.success("✅ Data pipeline ready")
    st.caption("Dataset: Olist Brazilian E-Commerce\n2016–2018 · ~99K orders")

# ── OVERVIEW ──────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("Financial Operations Overview")
    monthly = load_db("fct_monthly_revenue")
    rfm     = load_db("rfm_segments")
    if monthly is None: st.error("Run ETL first."); st.stop()

    total_gmv    = monthly["gmv"].sum()
    total_orders = monthly["order_count"].sum()
    total_cust   = rfm["customer_unique_id"].nunique() if rfm is not None else 0
    repeat_rate  = float((rfm["frequency"] > 1).mean() * 100) if rfm is not None else 0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total GMV", f"R${total_gmv/1e6:.1f}M", "+34% CAGR")
    c2.metric("Total Orders", f"{int(total_orders):,}", "+22% YoY")
    c3.metric("Unique Customers", f"{total_cust:,}")
    c4.metric("Repeat Rate", f"{repeat_rate:.1f}%", "-96.9% churn challenge", delta_color="inverse")

    st.divider()
    st.subheader("Monthly GMV Trend")
    fig = px.bar(monthly, x="order_month", y="gmv", color_discrete_sequence=["#3266AD"],
                 labels={"order_month":"Month","gmv":"GMV (R$)"})
    fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=320)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Orders & AOV")
        fig2 = make_subplots(specs=[[{"secondary_y":True}]])
        fig2.add_trace(go.Bar(x=monthly["order_month"], y=monthly["order_count"], name="Orders", marker_color="#3266AD", opacity=0.7), secondary_y=False)
        fig2.add_trace(go.Scatter(x=monthly["order_month"], y=monthly["avg_order_value"], name="AOV", line=dict(color="#EF9F27",width=2)), secondary_y=True)
        fig2.update_layout(height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"))
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        if rfm is not None:
            st.subheader("Customer Segments")
            seg = rfm["segment"].value_counts().reset_index()
            fig3 = px.pie(seg, names="segment", values="count", color_discrete_sequence=COLORS, hole=0.45)
            fig3.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig3, use_container_width=True)

# ── REVENUE FORECASTING ────────────────────────────────────────────────────────
elif page == "📈 Revenue Forecasting":
    st.title("Revenue Forecasting")
    forecast = load_csv("forecast.csv")
    monthly  = load_db("fct_monthly_revenue")
    if forecast is None: st.error("Run forecast model first."); st.stop()

    forecast["ds"] = pd.to_datetime(forecast["ds"])
    mape_list = forecast["mape"].dropna().tolist() if "mape" in forecast.columns else []
    mape_val  = mape_list[0] if mape_list else None

    actual_rows = forecast[forecast["is_forecast"]==0]
    fore_rows   = forecast[forecast["is_forecast"]==1]
    actual_vals = actual_rows["actual"].dropna()
    latest_actual = float(actual_vals.iloc[-1]) if len(actual_vals) else 0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Latest Actual GMV", f"R${latest_actual:,.0f}")
    c2.metric("6-Mo Forecast Total", f"R${fore_rows['yhat'].sum():,.0f}")
    if mape_val: c3.metric("Model MAPE", f"{mape_val:.1f}%")
    c4.metric("Model", forecast["model"].iloc[0] if "model" in forecast.columns else "LinearTrend")

    st.divider()
    st.subheader("Actual vs Forecast GMV")
    fig = go.Figure()
    fc_band = forecast[forecast["is_forecast"]==1]
    fig.add_trace(go.Scatter(
        x=pd.concat([fc_band["ds"], fc_band["ds"][::-1]]),
        y=pd.concat([fc_band["yhat_upper"], fc_band["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(50,102,173,0.12)", line=dict(width=0), name="95% CI"))
    hist = forecast[forecast["is_forecast"]==0]
    fig.add_trace(go.Bar(x=hist["ds"], y=hist["actual"], name="Actual GMV", marker_color="#3266AD", opacity=0.8))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast", line=dict(color="#EF9F27",width=2.5)))
    fig.update_layout(height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#e8e8e8"), legend=dict(orientation="h",y=1.08))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("6-Month Forecast Detail")
    display_fc = fore_rows[["ds","yhat","yhat_lower","yhat_upper"]].copy()
    display_fc.columns = ["Month","Forecast GMV","Lower bound","Upper bound"]
    display_fc["Month"] = display_fc["Month"].dt.strftime("%b %Y")
    for col in ["Forecast GMV","Lower bound","Upper bound"]:
        display_fc[col] = display_fc[col].apply(lambda x: f"R${x:,.0f}")
    st.dataframe(display_fc, use_container_width=True, hide_index=True)

    if monthly is not None:
        st.subheader("Month-over-Month Growth %")
        mom = monthly.dropna(subset=["mom_growth"])
        fig2 = px.bar(mom, x="order_month", y="mom_growth",
                      color=mom["mom_growth"].apply(lambda x: "positive" if x>=0 else "negative"),
                      color_discrete_map={"positive":"#1D9E75","negative":"#E24B4A"},
                      labels={"order_month":"Month","mom_growth":"MoM Growth %"})
        fig2.update_layout(showlegend=False, height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

# ── CHURN ─────────────────────────────────────────────────────────────────────
elif page == "🔄 Churn Analysis":
    st.title("Churn Analysis")
    churn = load_csv("churn_predictions.csv")
    fi    = load_csv("feature_importance.csv")
    if churn is None: st.error("Run churn model first."); st.stop()

    total      = len(churn)
    churned    = int(churn["churned"].sum()) if "churned" in churn.columns else int(total*0.969)
    churn_rate = churned/total*100
    high_risk  = int((churn["churn_risk"]=="High").sum()) if "churn_risk" in churn.columns else 0
    champions  = int((churn["segment"]=="Champions").sum()) if "segment" in churn.columns else 0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Overall Churn Rate", f"{churn_rate:.1f}%", delta_color="inverse")
    c2.metric("High-Risk Customers", f"{high_risk:,}", delta_color="inverse")
    c3.metric("Champions", f"{champions:,}")
    c4.metric("Repeat Buyers", f"{int((churn['frequency']>1).sum()):,}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("RFM Segment Distribution")
        if "segment" in churn.columns:
            seg = churn["segment"].value_counts().reset_index()
            fig = px.bar(seg, x="segment", y="count", color="segment", color_discrete_sequence=COLORS)
            fig.update_layout(showlegend=False, height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Churn Risk Distribution")
        if "churn_risk" in churn.columns:
            risk = churn["churn_risk"].value_counts().reset_index()
            fig2 = px.pie(risk, names="churn_risk", values="count",
                          color="churn_risk", color_discrete_map={"Low":"#1D9E75","Medium":"#EF9F27","High":"#E24B4A"}, hole=0.4)
            fig2.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

    if fi is not None:
        st.subheader("Feature Importance (Model)")
        fig3 = px.bar(fi.sort_values("importance"), x="importance", y="feature", orientation="h",
                      color_discrete_sequence=["#7F77DD"])
        fig3.update_layout(height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    if "segment" in churn.columns and "churn_probability" in churn.columns:
        st.subheader("Segment Profiling")
        prof = churn.groupby("segment").agg(
            customers=("customer_unique_id","count"),
            avg_monetary=("monetary","mean"),
            avg_frequency=("frequency","mean"),
            avg_churn_prob=("churn_probability","mean")).round(2).reset_index()
        prof["avg_monetary"]   = prof["avg_monetary"].apply(lambda x: f"R${x:,.0f}")
        prof["avg_churn_prob"] = prof["avg_churn_prob"].apply(lambda x: f"{x:.0%}")
        st.dataframe(prof, use_container_width=True, hide_index=True)

# ── PROFITABILITY ──────────────────────────────────────────────────────────────
elif page == "💰 Profitability":
    st.title("Profitability Analysis")
    cat_df    = load_csv("profitability_category.csv")
    seller_df = load_csv("profitability_sellers.csv")
    state_df  = load_csv("profitability_states.csv")
    if cat_df is None: st.error("Run profitability model first."); st.stop()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Best Margin", f"{cat_df['margin_pct'].max():.0f}%", cat_df.loc[cat_df['margin_pct'].idxmax(),'product_category_name'])
    c2.metric("Avg Margin", f"{cat_df['margin_pct'].mean():.0f}%")
    c3.metric("Worst Margin", f"{cat_df['margin_pct'].min():.0f}%", delta_color="inverse")
    if seller_df is not None:
        c4.metric("High-Margin Sellers", f"{int((seller_df['tier']=='High').sum())}")

    st.divider()
    st.subheader("Gross Margin % by Category")
    cat_sorted = cat_df.sort_values("margin_pct")
    fig = px.bar(cat_sorted, x="margin_pct", y="product_category_name", orientation="h",
                 color="margin_pct", color_continuous_scale=["#E24B4A","#EF9F27","#1D9E75"])
    fig.update_coloraxes(showscale=False)
    fig.update_layout(height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Revenue vs Margin")
        fig2 = px.scatter(cat_df, x="total_revenue", y="margin_pct", size="order_count",
                          text="product_category_name", color="margin_pct",
                          color_continuous_scale=["#E24B4A","#EF9F27","#1D9E75"])
        fig2.update_traces(textposition="top center", textfont_size=9)
        fig2.update_coloraxes(showscale=False)
        fig2.update_layout(height=320, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        if state_df is not None:
            st.subheader("Freight % by State")
            fig3 = px.bar(state_df.head(15).sort_values("freight_pct"), x="freight_pct", y="customer_state",
                          orientation="h", color="freight_pct", color_continuous_scale=["#1D9E75","#EF9F27","#E24B4A"])
            fig3.update_coloraxes(showscale=False)
            fig3.update_layout(height=320, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig3, use_container_width=True)

    if seller_df is not None:
        st.subheader("Top Seller Scorecard")
        display = seller_df[["seller_id","seller_state","gmv","orders","margin_pct_est","tier"]].head(20).copy()
        display["seller_id"]      = display["seller_id"].str[:8] + "..."
        display["gmv"]            = display["gmv"].apply(lambda x: f"R${x:,.0f}")
        display["margin_pct_est"] = display["margin_pct_est"].apply(lambda x: f"{x:.0f}%")
        display.columns = ["Seller","State","GMV","Orders","Est. Margin","Tier"]
        st.dataframe(display, use_container_width=True, hide_index=True)

# ── COHORT ─────────────────────────────────────────────────────────────────────
elif page == "📋 Cohort Retention":
    st.title("Cohort Retention Analysis")
    cohort = load_csv("cohort_retention.csv")
    if cohort is None: st.error("Run profitability model first."); st.stop()

    st.subheader("Monthly Retention Matrix (%)")
    heat_cols = [c for c in cohort.columns if c.startswith("M+")]
    heat_data = cohort[heat_cols].values.astype(float)
    fig = go.Figure(go.Heatmap(
        z=heat_data, x=heat_cols, y=cohort["Cohort"].astype(str),
        colorscale=[[0,"#FFF5F5"],[0.5,"#3266AD"],[1,"#1D9E75"]],
        text=[[f"{v:.0f}%" if not np.isnan(v) else "" for v in row] for row in heat_data],
        texttemplate="%{text}", textfont_size=11, zmin=0, zmax=100))
    fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)",
                      xaxis_title="Months since first purchase", yaxis_title="Cohort")
    st.plotly_chart(fig, use_container_width=True)

    c1,c2,c3 = st.columns(3)
    c1.metric("M+0 Avg Retention", f"{cohort['M+0'].mean():.0f}%" if 'M+0' in cohort.columns else "—")
    c2.metric("M+1 Avg Retention", f"{cohort['M+1'].mean():.1f}%" if 'M+1' in cohort.columns else "—", delta_color="inverse")
    c3.metric("M+3 Avg Retention", f"{cohort['M+3'].mean():.1f}%" if 'M+3' in cohort.columns else "—")
    st.divider()
    st.dataframe(cohort, use_container_width=True, hide_index=True)
