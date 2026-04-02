"""
etl.py  —  Load, clean, merge and persist all Olist tables.
Run:  python src/etl.py
Output: data/olist.db  (SQLite)
"""
import pandas as pd
import sqlite3
from pathlib import Path

RAW = Path(__file__).parent.parent / "data" / "raw"
DB  = str(Path(__file__).parent.parent / "data" / "olist.db")


# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
def load_raw():
    def read(name):
        return pd.read_csv(RAW / f"olist_{name}_dataset.csv")

    orders    = pd.read_csv(RAW / "olist_orders_dataset.csv",
                            parse_dates=["order_purchase_timestamp",
                                         "order_delivered_customer_date"])
    items     = read("order_items")
    customers = read("customers")
    payments  = read("order_payments")
    reviews   = read("order_reviews")
    products  = read("products")
    sellers   = read("sellers")
    return orders, items, customers, payments, reviews, products, sellers


# ─────────────────────────────────────────────
# 2. TRANSFORM
# ─────────────────────────────────────────────
def build_fact_orders(orders, items, customers, payments, reviews):
    # Aggregate order items
    items_agg = (items
        .groupby("order_id")
        .agg(item_count=("order_item_id","count"),
             total_price=("price","sum"),
             total_freight=("freight_value","sum"),
             seller_id=("seller_id","first"))
        .reset_index())

    # Aggregate payments
    pay_agg = (payments
        .groupby("order_id")
        .agg(payment_value=("payment_value","sum"),
             payment_type=("payment_type","first"),
             installments=("payment_installments","max"))
        .reset_index())

    # Average review score
    rev_agg = (reviews
        .groupby("order_id")["review_score"]
        .mean()
        .reset_index()
        .rename(columns={"review_score":"avg_review_score"}))

    fact = (orders
        .merge(customers[["customer_id","customer_unique_id",
                          "customer_state","customer_city"]], on="customer_id")
        .merge(items_agg, on="order_id", how="left")
        .merge(pay_agg,   on="order_id", how="left")
        .merge(rev_agg,   on="order_id", how="left"))

    # Derived columns
    fact["order_month"]   = fact["order_purchase_timestamp"].dt.to_period("M").astype(str)
    fact["order_year"]    = fact["order_purchase_timestamp"].dt.year
    fact["order_quarter"] = fact["order_purchase_timestamp"].dt.quarter
    fact["delivery_days"] = (
        fact["order_delivered_customer_date"] -
        fact["order_purchase_timestamp"]
    ).dt.days
    fact["freight_ratio"] = (
        fact["total_freight"] / fact["total_price"].replace(0, pd.NA)
    )
    fact["is_delivered"]  = (fact["order_status"] == "delivered").astype(int)

    # Keep only meaningful statuses
    return fact[fact["order_status"].isin(["delivered","shipped"])].copy()


def build_rfm(fact, snapshot="2018-08-31"):
    snap = pd.Timestamp(snapshot)
    rfm = (fact[fact["is_delivered"] == 1]
        .groupby("customer_unique_id")
        .agg(
            last_order=("order_purchase_timestamp","max"),
            frequency=("order_id","nunique"),
            monetary=("payment_value","sum"))
        .reset_index())
    rfm["recency_days"] = (snap - rfm["last_order"]).dt.days
    rfm["clv_proxy"]    = rfm["monetary"] * rfm["frequency"] / (rfm["recency_days"] + 1)

    # RFM scores (1–5)
    rfm["r_score"] = pd.qcut(rfm["recency_days"], 5, labels=[5,4,3,2,1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    rfm["rfm_score"]= rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    def segment(row):
        r, f = row["r_score"], row["f_score"]
        if r >= 4 and f >= 4: return "Champions"
        if r >= 3 and f >= 3: return "Loyal"
        if r >= 3 and f <= 2: return "Potential Loyalists"
        if r <= 2 and f >= 3: return "At-Risk"
        if r == 1 and f == 1: return "Lost"
        return "Others"

    rfm["segment"] = rfm.apply(segment, axis=1)
    return rfm


def build_monthly_revenue(fact):
    df = (fact[fact["is_delivered"] == 1]
        .groupby("order_month")
        .agg(
            order_count=("order_id","nunique"),
            unique_customers=("customer_unique_id","nunique"),
            gmv=("payment_value","sum"),
            avg_order_value=("payment_value","mean"),
            avg_review=("avg_review_score","mean"),
            avg_freight=("total_freight","mean"))
        .reset_index()
        .sort_values("order_month"))
    df["gmv_lag1"]     = df["gmv"].shift(1)
    df["mom_growth"]   = (df["gmv"] - df["gmv_lag1"]) / df["gmv_lag1"] * 100
    df["gmv_3m_avg"]   = df["gmv"].rolling(3).mean()
    return df


def build_category_profitability(items, products):
    merged = items.merge(products[["product_id","product_category_name"]], on="product_id", how="left")
    cat = (merged
        .groupby("product_category_name")
        .agg(total_revenue=("price","sum"),
             total_freight=("freight_value","sum"),
             order_count=("order_id","nunique"))
        .reset_index())
    cat["net_revenue"] = cat["total_revenue"] - cat["total_freight"]
    cat["margin_pct"]  = (cat["net_revenue"] / cat["total_revenue"] * 100).round(1)
    return cat.sort_values("total_revenue", ascending=False)


# ─────────────────────────────────────────────
# 3. SAVE
# ─────────────────────────────────────────────
def save(df, table):
    con = sqlite3.connect(DB)
    df.to_sql(table, con, if_exists="replace", index=False)
    con.close()
    print(f"  ✓  {table:30s}  {len(df):>8,} rows")


def run_etl():
    print("Loading raw CSVs...")
    orders, items, customers, payments, reviews, products, sellers = load_raw()

    print("Building tables...")
    fact   = build_fact_orders(orders, items, customers, payments, reviews)
    rfm    = build_rfm(fact)
    monthly= build_monthly_revenue(fact)
    cat_p  = build_category_profitability(items, products)

    print("Saving to SQLite...")
    save(fact,     "fact_orders")
    save(rfm,      "rfm_segments")
    save(monthly,  "fct_monthly_revenue")
    save(cat_p,    "fct_category_profitability")
    save(customers,"dim_customers")
    save(sellers,  "dim_sellers")
    save(products, "dim_products")
    print(f"\nETL complete → {DB}")


if __name__ == "__main__":
    run_etl()
