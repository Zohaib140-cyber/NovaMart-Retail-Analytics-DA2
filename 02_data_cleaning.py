"""
NovaMart Retail Analytics – Data Cleaning Script
"""
import pandas as pd
import numpy as np
import warnings

import os
os.makedirs("data", exist_ok=True)
os.makedirs("charts", exist_ok=True)

warnings.filterwarnings("ignore")

print("=" * 60)
print("  NOVAMART – DATA CLEANING PIPELINE")
print("=" * 60)

df = pd.read_csv("data/novamart_raw.csv")
print(f"\n[BEFORE] Shape: {df.shape}")
print(f"[BEFORE] Missing values:\n{df.isnull().sum()}")
print(f"[BEFORE] Duplicates: {df.duplicated().sum()}")
print(f"[BEFORE] Negative Qty: {(df['Quantity']<=0).sum()}")

# ── STEP 1: Remove duplicates ──────────────────────────────
before = len(df)
df.drop_duplicates(inplace=True)
print(f"\n[STEP 1] Duplicates removed: {before - len(df)}")

# ── STEP 2: Fix Purchase_Date & remove future dates ────────
df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"], errors="coerce")
cutoff = pd.Timestamp("2026-01-01")
invalid_dates = (df["Purchase_Date"] > cutoff) | (df["Purchase_Date"].isna())
print(f"[STEP 2] Invalid/future dates removed: {invalid_dates.sum()}")
df = df[~invalid_dates].copy()
df["Purchase_Date"] = df["Purchase_Date"].dt.strftime("%Y-%m-%d %H:%M:%S")

# ── STEP 3: Drop rows where core business fields are null ──
core_cols = ["Transaction_ID","Product_Category","Product_Name","Purchase_Date"]
before = len(df)
df.dropna(subset=core_cols, inplace=True)
print(f"[STEP 3] Rows removed (null core fields): {before - len(df)}")

# ── STEP 4: Standardize City names ────────────────────────
city_mapping = {
    "karachi":"Karachi","KARACHI":"Karachi","karachii":"Karachi",
    "lahore":"Lahore","LAHORE":"Lahore","lahor":"Lahore",
    "islamabad":"Islamabad","ISLAMABAD":"Islamabad",
    "rawalpindi":"Rawalpindi","RAWALPINDI":"Rawalpindi","rawal pindi":"Rawalpindi",
    "faisalabad":"Faisalabad","FAISALABAD":"Faisalabad",
    "multan":"Multan","MULTAN":"Multan",
    "peshawar":"Peshawar","PESHAWAR":"Peshawar",
    "quetta":"Quetta","QUETTA":"Quetta",
    "hyderabad":"Hyderabad","sialkot":"Sialkot",
}
df["Customer_City"] = df["Customer_City"].str.strip()
df["Customer_City"] = df["Customer_City"].replace(city_mapping)
print(f"[STEP 4] City standardization complete. Unique cities: {df['Customer_City'].nunique()}")

# ── STEP 5: Standardize Payment Method ─────────────────────
pay_mapping = {
    "cash":"Cash","CASH":"Cash",
    "credit card":"Credit Card","creditcard":"Credit Card","Credit card":"Credit Card",
    "debit card":"Debit Card","DEBIT CARD":"Debit Card",
    "jazzcash":"JazzCash","jazz cash":"JazzCash","Jazz Cash":"JazzCash",
    "easypaisa":"EasyPaisa","easy paisa":"EasyPaisa",
    "bank transfer":"Bank Transfer","BANK TRANSFER":"Bank Transfer",
}
df["Payment_Method"] = df["Payment_Method"].str.strip()
df["Payment_Method"] = df["Payment_Method"].replace(pay_mapping)
# Fill remaining nulls with mode
mode_pay = df["Payment_Method"].mode()[0]
df = df.assign(Payment_Method=df["Payment_Method"].fillna(mode_pay))
print(f"[STEP 5] Payment method standardized. Unique: {df['Payment_Method'].nunique()}")

# ── STEP 6: Fix Quantity (remove <=0) ──────────────────────
before = len(df)
df = df[df["Quantity"] > 0].copy()
print(f"[STEP 6] Rows with invalid Quantity removed: {before - len(df)}")

# ── STEP 7: Impute Unit_Price where missing ─────────────────
# Use median price per product
product_median_price = df.groupby("Product_Name")["Unit_Price"].transform("median")
df["Unit_Price"] = df["Unit_Price"].fillna(product_median_price)
df["Unit_Price"] = df["Unit_Price"].fillna(df["Unit_Price"].median())  # fallback

# Recalculate Total_Amount where missing
df["Discount_Pct"] = df["Discount_Pct"].fillna(0)
mask = df["Total_Amount"].isna()
df.loc[mask, "Total_Amount"] = (
    df.loc[mask, "Unit_Price"] * df.loc[mask, "Quantity"] * (1 - df.loc[mask, "Discount_Pct"]/100)
).round(2)
print(f"[STEP 7] Unit_Price & Total_Amount imputed where missing")

# ── STEP 8: Fill remaining categorical nulls ────────────────
df = df.assign(
    Customer_ID=df["Customer_ID"].fillna("CUST-UNKNOWN"),
    Customer_Gender=df["Customer_Gender"].fillna("Unknown"),
    Customer_Rating=pd.to_numeric(df["Customer_Rating"], errors="coerce").fillna(4)
)
print(f"[STEP 8] Categorical nulls filled")

# ── STEP 9: Data type correction ───────────────────────────
df["Purchase_Date"]    = pd.to_datetime(df["Purchase_Date"])
df["Quantity"]         = df["Quantity"].astype(int)
df["Unit_Price"]       = df["Unit_Price"].round(2)
df["Total_Amount"]     = df["Total_Amount"].round(2)
df["Discount_Pct"]     = df["Discount_Pct"].astype(int)
df["Customer_Rating"]  = pd.to_numeric(df["Customer_Rating"], errors="coerce").fillna(4).round(0).astype(int)
print(f"[STEP 9] Data types corrected")

# ── STEP 10: Add derived columns for EDA ───────────────────
df["Year"]         = df["Purchase_Date"].dt.year
df["Month"]        = df["Purchase_Date"].dt.month
df["Month_Name"]   = df["Purchase_Date"].dt.strftime("%b")
df["Day_of_Week"]  = df["Purchase_Date"].dt.strftime("%A")
df["Hour"]         = df["Purchase_Date"].dt.hour
df["Quarter"]      = df["Purchase_Date"].dt.quarter
df["Is_Weekend"]   = df["Purchase_Date"].dt.dayofweek.isin([5,6]).astype(int)
print(f"[STEP 10] Derived columns added")

# ── Save cleaned dataset ────────────────────────────────────
df.to_csv("data/novamart_cleaned.csv", index=False)

print("\n" + "=" * 60)
print("  CLEANING COMPLETE – AFTER STATISTICS")
print("=" * 60)
print(f"[AFTER] Shape: {df.shape}")
print(f"[AFTER] Missing values:\n{df.isnull().sum()}")
print(f"[AFTER] Duplicates: {df.duplicated().sum()}")
print(f"\nTotal Revenue: PKR {df['Total_Amount'].sum():,.2f}")
print(f"Total Orders:  {len(df):,}")
print(f"Avg Order Val: PKR {df['Total_Amount'].mean():,.2f}")
print(f"Date Range:    {df['Purchase_Date'].min().date()} → {df['Purchase_Date'].max().date()}")
# (Fix applied inline above – rerun with patch)
