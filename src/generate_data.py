"""
generate_data.py
Generates realistic synthetic Olist-like data so the project runs
immediately without requiring the Kaggle download.
Replace with real CSVs when available — the ETL pipeline is identical.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import random, string

RAW = Path(__file__).parent.parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
random.seed(42)

CATEGORIES = [
    "health_beauty", "bed_bath_table", "sports_leisure",
    "computers_accessories", "furniture_decor", "watches_gifts",
    "toys", "office_furniture", "garden_tools", "auto"
]
CAT_BASE_PRICE = {
    "health_beauty": 55, "bed_bath_table": 110, "sports_leisure": 90,
    "computers_accessories": 220, "furniture_decor": 180, "watches_gifts": 160,
    "toys": 65, "office_furniture": 140, "garden_tools": 95, "auto": 130
}
STATES = ["SP","RJ","MG","RS","PR","SC","BA","GO","PE","CE",
          "PA","MA","AM","MT","MS","ES","RN","PB","RO","AC"]
STATE_FREIGHT_MULT = {
    "SP":1.0,"RJ":1.1,"MG":1.2,"RS":1.4,"PR":1.3,"SC":1.35,
    "BA":1.6,"GO":1.5,"PE":1.7,"CE":1.75,"PA":2.0,"MA":1.9,
    "AM":2.2,"MT":1.8,"MS":1.65,"ES":1.3,"RN":1.85,"PB":1.8,
    "RO":2.1,"AC":2.4
}
PAYMENT_TYPES = ["credit_card","boleto","voucher","debit_card"]
N_CUSTOMERS = 96096
N_SELLERS   = 3095
N_ORDERS    = 12000   # ~99K in real dataset; use 12K for fast generation

def uid(n=8): return ''.join(random.choices(string.hexdigits[:16], k=n))

print("Generating customers...")
customer_ids  = [uid(32) for _ in range(N_CUSTOMERS)]
customer_uids = [uid(32) for _ in range(N_CUSTOMERS)]
p_raw = [.35,.15,.10,.07,.06,.05,.05,.03,.03,.03,.01,.01,.01,.01,.01,.01,.01,.01,.005,.005]
p_arr = np.array(p_raw); p_arr = p_arr / p_arr.sum()
cust_states   = rng.choice(STATES, N_CUSTOMERS, p=p_arr)
customers = pd.DataFrame({
    "customer_id":         customer_ids,
    "customer_unique_id":  customer_uids,
    "customer_zip_code_prefix": rng.integers(10000,99999,N_CUSTOMERS),
    "customer_city":       ["city_"+s.lower() for s in cust_states],
    "customer_state":      cust_states
})
customers.to_csv(RAW/"olist_customers_dataset.csv", index=False)

print("Generating sellers...")
seller_ids    = [uid(32) for _ in range(N_SELLERS)]
sell_states   = rng.choice(STATES, N_SELLERS)
sellers = pd.DataFrame({
    "seller_id":           seller_ids,
    "seller_zip_code_prefix": rng.integers(10000,99999,N_SELLERS),
    "seller_city":         ["city_"+s.lower() for s in sell_states],
    "seller_state":        sell_states
})
sellers.to_csv(RAW/"olist_sellers_dataset.csv", index=False)

print("Generating products...")
product_ids = [uid(32) for _ in range(4000)]
prod_cats   = rng.choice(CATEGORIES, 4000)
products = pd.DataFrame({
    "product_id":                product_ids,
    "product_category_name":     prod_cats,
    "product_weight_g":          rng.integers(100, 30000, 4000),
    "product_length_cm":         rng.integers(10, 100, 4000),
    "product_height_cm":         rng.integers(5, 50, 4000),
    "product_width_cm":          rng.integers(10, 80, 4000),
})
products.to_csv(RAW/"olist_products_dataset.csv", index=False)

print("Generating orders with realistic seasonality...")
# Dates 2016-10 through 2018-08 with monthly volume seasonality
def seasonal_weight(dt):
    month = dt.month
    base  = [0.6,0.65,0.8,0.85,0.9,0.95,1.0,1.05,0.95,1.0,2.2,1.4]
    year_mult = 1.0 if dt.year == 2016 else (1.35 if dt.year == 2017 else 1.75)
    return base[month-1] * year_mult

date_range  = pd.date_range("2016-10-01","2018-08-31",freq="D")
day_weights = np.array([seasonal_weight(d) for d in date_range])
day_weights /= day_weights.sum()
order_dates = rng.choice(date_range, N_ORDERS, p=day_weights)
order_dates = pd.to_datetime(order_dates)

cust_sample = rng.choice(customer_ids, N_ORDERS)
statuses    = rng.choice(
    ["delivered","shipped","canceled","unavailable","invoiced"],
    N_ORDERS, p=[0.965,0.015,0.011,0.005,0.004]
)
delivery_days = rng.integers(3,40,N_ORDERS)
delivered_at  = [
    (d + pd.Timedelta(days=int(dd))).isoformat() if s=="delivered" else None
    for d,dd,s in zip(order_dates,delivery_days,statuses)
]
order_ids = [uid(32) for _ in range(N_ORDERS)]
orders = pd.DataFrame({
    "order_id":                   order_ids,
    "customer_id":                cust_sample,
    "order_status":               statuses,
    "order_purchase_timestamp":   order_dates.astype(str),
    "order_approved_at":          [(pd.Timestamp(d)+pd.Timedelta(hours=rng.integers(1,24))).isoformat() for d in order_dates],
    "order_delivered_carrier_date": [(pd.Timestamp(d)+pd.Timedelta(days=rng.integers(1,5))).isoformat() for d in order_dates],
    "order_delivered_customer_date": delivered_at,
    "order_estimated_delivery_date": [(pd.Timestamp(d)+pd.Timedelta(days=rng.integers(10,35))).isoformat() for d in order_dates],
})
orders.to_csv(RAW/"olist_orders_dataset.csv", index=False)

print("Generating order items...")
# Build lookup maps for speed
prod_cat_map   = dict(zip(products["product_id"], products["product_category_name"]))
seller_state_map = dict(zip(sellers["seller_id"], sellers["seller_state"]))

items_rows = []
for oid in order_ids:
    n_items = int(rng.choice([1,2,3,4], p=[0.75,0.16,0.06,0.03]))
    for rank in range(1, n_items+1):
        pid   = rng.choice(product_ids)
        cat   = prod_cat_map.get(pid, "health_beauty")
        base  = CAT_BASE_PRICE.get(cat, 100)
        price = round(float(rng.normal(base, base*0.3)), 2)
        price = max(9.9, price)
        sid   = rng.choice(seller_ids)
        state = seller_state_map.get(sid, "SP")
        freight = round(price * 0.10 * STATE_FREIGHT_MULT.get(state, 1.5), 2)
        items_rows.append({
            "order_id": oid, "order_item_id": rank,
            "product_id": pid, "seller_id": sid,
            "shipping_limit_date": "2018-01-01",
            "price": price, "freight_value": freight
        })
order_items = pd.DataFrame(items_rows)
order_items.to_csv(RAW/"olist_order_items_dataset.csv", index=False)

print("Generating payments...")
pay_rows2 = []
for oid in order_ids:
    its = order_items[order_items.order_id==oid]
    total = round(float(its.price.sum() + its.freight_value.sum()),2)
    ptype = rng.choice(PAYMENT_TYPES, p=[0.73,0.19,0.05,0.03])
    inst  = int(rng.choice([1,2,3,6,10,12], p=[0.40,0.15,0.15,0.12,0.10,0.08]))
    pay_rows2.append({
        "order_id":oid,"payment_sequential":1,
        "payment_type":ptype,"payment_installments":inst,
        "payment_value":total
    })
payments = pd.DataFrame(pay_rows2)
payments.to_csv(RAW/"olist_order_payments_dataset.csv", index=False)

print("Generating reviews...")
review_rows = []
for oid, stat in zip(order_ids, statuses):
    if stat == "delivered" and rng.random() < 0.78:
        score = int(rng.choice([1,2,3,4,5], p=[0.06,0.05,0.09,0.20,0.60]))
        review_rows.append({"review_id":uid(32),"order_id":oid,"review_score":score,
            "review_comment_title":"","review_comment_message":"",
            "review_creation_date":"2018-01-01","review_answer_timestamp":"2018-01-02"})
reviews_df = pd.DataFrame(review_rows)
reviews_df.to_csv(RAW/"olist_order_reviews_dataset.csv", index=False)

print(f"\nAll synthetic data generated in {RAW}")
print(f"  orders:    {N_ORDERS:,}")
print(f"  customers: {N_CUSTOMERS:,}")
print(f"  sellers:   {N_SELLERS:,}")
print(f"  items:     {len(order_items):,}")
