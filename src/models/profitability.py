"""
profitability.py  —  Margin analysis by category, seller, and state.
Run:  python src/models/profitability.py
Output: data/processed/profitability_*.csv
"""
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path

DB  = str(Path(__file__).parent.parent.parent / "data" / "olist.db")
OUT = Path(__file__).parent.parent.parent / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def load_data():
    con = sqlite3.connect(DB)
    fact     = pd.read_sql("SELECT * FROM fact_orders WHERE is_delivered=1", con)
    cat_prof = pd.read_sql("SELECT * FROM fct_category_profitability", con)
    sellers  = pd.read_sql("SELECT * FROM dim_sellers", con)
    con.close()
    return fact, cat_prof, sellers


def category_analysis(cat_prof):
    """Category-level profitability."""
    df = cat_prof.copy()
    df["rank"]           = df["total_revenue"].rank(ascending=False).astype(int)
    df["revenue_share"]  = (df["total_revenue"] / df["total_revenue"].sum() * 100).round(1)
    df = df.sort_values("margin_pct", ascending=False)
    return df


def seller_scorecard(fact, sellers):
    """Top seller profitability scorecard."""
    df = (fact
        .groupby("seller_id")
        .agg(
            gmv=("payment_value","sum"),
            orders=("order_id","nunique"),
            avg_review=("avg_review_score","mean"),
            avg_freight_ratio=("freight_ratio","mean"),
            customers=("customer_unique_id","nunique")
        )
        .reset_index())

    df = df.merge(sellers[["seller_id","seller_state"]], on="seller_id", how="left")
    df["avg_order_value"] = df["gmv"] / df["orders"]
    df["freight_cost_est"]= df["gmv"] * df["avg_freight_ratio"].fillna(0.12)
    df["net_revenue_est"] = df["gmv"] - df["freight_cost_est"]
    df["margin_pct_est"]  = (df["net_revenue_est"] / df["gmv"] * 100).round(1)

    # Tier
    df["tier"] = pd.cut(df["margin_pct_est"],
        bins=[-np.inf, 20, 45, np.inf],
        labels=["Low", "Medium", "High"])

    return df.sort_values("gmv", ascending=False).head(50)


def state_freight_analysis(fact):
    """Average freight burden by customer state."""
    df = (fact
        .groupby("customer_state")
        .agg(
            total_orders=("order_id","nunique"),
            avg_freight=("total_freight","mean"),
            avg_order_value=("payment_value","mean"),
            avg_delivery_days=("delivery_days","mean")
        )
        .reset_index())
    df["freight_pct"] = (df["avg_freight"] / df["avg_order_value"] * 100).round(1)
    return df.sort_values("freight_pct", ascending=False)


def cohort_retention(fact):
    """Monthly cohort retention matrix."""
    df = fact[["customer_unique_id","order_month","order_id"]].copy()
    df["order_month"] = pd.to_datetime(df["order_month"])

    # First purchase month per customer
    first = (df.groupby("customer_unique_id")["order_month"]
               .min().reset_index()
               .rename(columns={"order_month":"cohort_month"}))
    df = df.merge(first, on="customer_unique_id")

    df["period"] = ((df["order_month"].dt.year  - df["cohort_month"].dt.year)*12 +
                    (df["order_month"].dt.month - df["cohort_month"].dt.month))

    cohort_size = (first.groupby("cohort_month")["customer_unique_id"]
                        .nunique().reset_index()
                        .rename(columns={"customer_unique_id":"cohort_size"}))

    retention = (df.groupby(["cohort_month","period"])["customer_unique_id"]
                   .nunique().reset_index()
                   .rename(columns={"customer_unique_id":"customers"}))
    retention = retention.merge(cohort_size, on="cohort_month")
    retention["retention_rate"] = (retention["customers"] /
                                   retention["cohort_size"] * 100).round(1)

    # Pivot to matrix — last 12 cohort months, periods 0–6
    pivot = (retention[retention["period"] <= 6]
             .pivot_table(index="cohort_month", columns="period",
                          values="retention_rate", aggfunc="first"))
    pivot.index = pivot.index.astype(str)
    pivot.columns = [f"M+{c}" for c in pivot.columns]
    return pivot.reset_index().rename(columns={"cohort_month":"Cohort"}).tail(12)


def run():
    print("Loading data...")
    fact, cat_prof, sellers = load_data()

    print("Running category analysis...")
    cat_df = category_analysis(cat_prof)
    cat_df.to_csv(OUT / "profitability_category.csv", index=False)
    print(f"  ✓ {len(cat_df)} categories")

    print("Running seller scorecard...")
    seller_df = seller_scorecard(fact, sellers)
    seller_df.to_csv(OUT / "profitability_sellers.csv", index=False)
    print(f"  ✓ Top {len(seller_df)} sellers scored")

    print("Running freight analysis by state...")
    state_df = state_freight_analysis(fact)
    state_df.to_csv(OUT / "profitability_states.csv", index=False)
    print(f"  ✓ {len(state_df)} states analyzed")

    print("Building cohort retention matrix...")
    cohort_df = cohort_retention(fact)
    cohort_df.to_csv(OUT / "cohort_retention.csv", index=False)
    print(f"  ✓ {len(cohort_df)} cohorts")

    print(f"\nAll profitability outputs saved → {OUT}")

    # Quick summary print
    print("\n--- Top 5 categories by margin ---")
    print(cat_df[["product_category_name","total_revenue","margin_pct"]].head(5).to_string(index=False))


if __name__ == "__main__":
    run()
