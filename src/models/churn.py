"""
churn.py  —  Customer churn prediction using XGBoost + SHAP.
Run:  python src/models/churn.py
Output: data/processed/churn_predictions.csv
        data/processed/feature_importance.csv
        data/processed/churn_model.joblib
"""
import pandas as pd
import numpy as np
import sqlite3, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

DB  = str(Path(__file__).parent.parent.parent / "data" / "olist.db")
OUT = Path(__file__).parent.parent.parent / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = ["recency_days", "frequency", "monetary",
            "avg_order_value", "freight_ratio", "clv_proxy"]


def load_rfm():
    con = sqlite3.connect(DB)
    rfm = pd.read_sql("SELECT * FROM rfm_segments", con)

    # Join additional features from fact_orders
    fact_agg = pd.read_sql("""
        SELECT customer_unique_id,
               AVG(payment_value)  AS avg_order_value,
               AVG(freight_ratio)  AS freight_ratio,
               AVG(avg_review_score) AS avg_review,
               COUNT(DISTINCT order_id) AS n_orders
        FROM fact_orders
        WHERE is_delivered = 1
        GROUP BY customer_unique_id
    """, con)
    con.close()

    df = rfm.merge(fact_agg, on="customer_unique_id", how="left")
    df["avg_order_value"] = df["avg_order_value"].fillna(df["monetary"])
    df["freight_ratio"]   = df["freight_ratio"].fillna(0.1)
    df["clv_proxy"]       = df.get("clv_proxy", df["monetary"] / (df["recency_days"] + 1))
    return df


def label_churn(df, threshold_days=180):
    """Churned = no order in last 180 days."""
    df = df.copy()
    df["churned"] = (df["recency_days"] > threshold_days).astype(int)
    return df


def build_and_evaluate(df):
    df   = label_churn(df)
    feat = [f for f in FEATURES if f in df.columns]

    X = df[feat].fillna(0)
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)

        # Feature importance
        importance = pd.DataFrame({
            "feature":   feat,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        importance = pd.DataFrame({
            "feature":   feat,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

    # Evaluation
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc    = roc_auc_score(y_test, y_prob)

    print(f"  AUC-ROC:  {auc:.4f}")
    print(f"  Churn rate in data: {y.mean():.1%}")
    print("\n" + classification_report(y_test, y_pred, target_names=["Retained","Churned"]))

    # Score full dataset
    df["churn_probability"] = model.predict_proba(X)[:, 1]
    df["churn_risk"] = pd.cut(
        df["churn_probability"],
        bins=[0, 0.4, 0.7, 1.0],
        labels=["Low", "Medium", "High"]
    )

    return model, df, importance, auc


def run():
    print("Loading RFM data...")
    df = load_rfm()
    print(f"  {len(df):,} customers")

    print("Training churn model...")
    model, scored_df, importance, auc = build_and_evaluate(df)

    # Save outputs
    scored_df.to_csv(OUT / "churn_predictions.csv", index=False)
    importance.to_csv(OUT / "feature_importance.csv", index=False)
    joblib.dump(model, OUT / "churn_model.joblib")

    print(f"\n  ✓ Predictions  → {OUT}/churn_predictions.csv")
    print(f"  ✓ Importance   → {OUT}/feature_importance.csv")
    print(f"  ✓ Model saved  → {OUT}/churn_model.joblib")

    # Segment summary
    seg = (scored_df.groupby("segment")
           .agg(customers=("customer_unique_id","count"),
                avg_monetary=("monetary","mean"),
                avg_churn_prob=("churn_probability","mean"))
           .round(2))
    print("\nSegment summary:")
    print(seg.to_string())

    return scored_df, importance, auc


if __name__ == "__main__":
    run()
