"""
=============================================================
NovaMart Retail Analytics – Complete EDA & Analysis Script
Task DA-2 | TEYZIX CORE Internship Program | June 2026
=============================================================
This script performs the complete Exploratory Data Analysis
on the cleaned NovaMart dataset. All charts are saved to
the /charts/ directory.

Requirements: pandas, numpy, matplotlib, seaborn, scikit-learn
Run: python 03_eda_complete.py
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")
import os

# ── Configuration ──────────────────────────────────────────
DATA_PATH   = "data/novamart_cleaned.csv"
CHARTS_DIR  = "charts/"
BRAND_GREEN = "#1B5E20"
BRAND_LIGHT = "#4CAF50"
ACCENT      = "#FF6F00"
PALETTE     = ["#1B5E20","#2E7D32","#388E3C","#43A047","#4CAF50","#66BB6A","#81C784","#A5D6A7"]
os.makedirs(CHARTS_DIR, exist_ok=True)

# ── Load Data ──────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["Purchase_Date"])
print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Date range: {df['Purchase_Date'].min().date()} → {df['Purchase_Date'].max().date()}")
print(f"Total Revenue: PKR {df['Total_Amount'].sum():,.2f}")
print(f"Avg Order Value: PKR {df['Total_Amount'].mean():,.2f}\n")

# Helper
MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
DAY_ORDER   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})

# ══════════════════════════════════════════════════════════
# ANALYSIS 1: Monthly Revenue Trend
# Objective: Identify seasonal demand cycles for planning
# ══════════════════════════════════════════════════════════
print("━" * 50)
print("ANALYSIS 1: Monthly Revenue & Order Trends")
monthly = (df.groupby(["Month","Month_Name"])["Total_Amount"]
           .agg(Revenue="sum", Orders="count").reset_index().sort_values("Month"))

print(f"Best Month : {monthly.loc[monthly['Revenue'].idxmax(),'Month_Name']} "
      f"(PKR {monthly['Revenue'].max():,.0f})")
print(f"Worst Month: {monthly.loc[monthly['Revenue'].idxmin(),'Month_Name']} "
      f"(PKR {monthly['Revenue'].min():,.0f})")
print(f"Revenue CoV (seasonality): {monthly['Revenue'].std()/monthly['Revenue'].mean()*100:.1f}%")

fig, ax1 = plt.subplots(figsize=(13,5))
bars = ax1.bar(monthly["Month_Name"], monthly["Revenue"]/1000, color=BRAND_GREEN, alpha=0.85, width=0.6, zorder=2)
ax2  = ax1.twinx()
ax2.plot(monthly["Month_Name"], monthly["Orders"], color=ACCENT, marker="o", linewidth=2.5, markersize=7, zorder=3, label="Order Count")
ax1.set_ylabel("Revenue (PKR '000)", fontsize=12, color=BRAND_GREEN)
ax2.set_ylabel("Number of Orders", fontsize=12, color=ACCENT)
ax1.set_title("NovaMart – Monthly Revenue & Order Volume (2025)", fontsize=14, fontweight="bold", pad=15)
ax1.tick_params(axis="y", colors=BRAND_GREEN); ax2.tick_params(axis="y", colors=ACCENT)
for bar in bars:
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{bar.get_height():.0f}K",
             ha="center", va="bottom", fontsize=8, color=BRAND_GREEN, fontweight="bold")
ax2.legend(loc="upper left")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}01_monthly_revenue.png", dpi=160, bbox_inches="tight")
plt.close()
print("→ Chart saved: 01_monthly_revenue.png\n")

# ══════════════════════════════════════════════════════════
# ANALYSIS 2: Top Products by Revenue
# Objective: Identify anchor SKUs for prioritization
# ══════════════════════════════════════════════════════════
print("━" * 50)
print("ANALYSIS 2: Top Products by Revenue")
top10 = df.groupby("Product_Name")["Total_Amount"].sum().sort_values(ascending=False).head(10)
print(top10.to_string())

fig, ax = plt.subplots(figsize=(11,6))
sorted_10 = top10.sort_values()
colors = [BRAND_GREEN if i >= 7 else BRAND_LIGHT if i >= 4 else "#A5D6A7" for i in range(len(sorted_10))]
bars = ax.barh(sorted_10.index, sorted_10.values/1000, color=colors)
for bar in bars:
    ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
            f"PKR {bar.get_width():.0f}K", va="center", fontsize=9, color="#333")
ax.set_xlabel("Total Revenue (PKR '000)"); ax.set_title("Top 10 Products by Total Revenue", fontsize=14, fontweight="bold", pad=15)
ax.set_xlim(0, sorted_10.max()/1000*1.18)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}02_top_products.png", dpi=160, bbox_inches="tight")
plt.close()
print("→ Chart saved: 02_top_products.png\n")

# ══════════════════════════════════════════════════════════
# ANALYSIS 3: Category Performance
# Objective: Benchmark categories across KPIs
# ══════════════════════════════════════════════════════════
print("━" * 50)
print("ANALYSIS 3: Category Performance")
cat_rev = df.groupby("Product_Category")["Total_Amount"].sum().sort_values(ascending=False)
cat_ord = df.groupby("Product_Category")["Transaction_ID"].count()
cat_avg = df.groupby("Product_Category")["Total_Amount"].mean()
cat_pct = (cat_rev / cat_rev.sum() * 100).round(1)
summary = pd.DataFrame({"Revenue_PKR":cat_rev,"Orders":cat_ord,"Avg_Order_Value":cat_avg.round(0),"Revenue_Pct":cat_pct})
print(summary.to_string())

fig, axes = plt.subplots(1,3,figsize=(16,5))
axes[0].barh(cat_rev.index[::-1], cat_rev.values[::-1]/1000, color=PALETTE[:len(cat_rev)])
axes[0].set_title("Revenue by Category\n(PKR '000)", fontweight="bold"); axes[0].set_xlabel("PKR '000")
axes[1].barh(cat_ord.sort_values().index, cat_ord.sort_values().values, color=PALETTE[:len(cat_ord)])
axes[1].set_title("Order Count\nby Category", fontweight="bold"); axes[1].set_xlabel("# Orders")
axes[2].barh(cat_avg.sort_values().index, cat_avg.sort_values().values, color=PALETTE[:len(cat_avg)])
axes[2].set_title("Avg Order Value\nby Category (PKR)", fontweight="bold"); axes[2].set_xlabel("PKR")
for ax in axes: ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.suptitle("Category Performance Dashboard", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}03_category_performance.png", dpi=160, bbox_inches="tight")
plt.close()
print("→ Chart saved: 03_category_performance.png\n")

# ══════════════════════════════════════════════════════════
# ANALYSIS 4: Revenue by City
# Objective: Geographic concentration and expansion analysis
# ══════════════════════════════════════════════════════════
print("━" * 50)
print("ANALYSIS 4: Revenue by City")
city_rev = df.groupby("Customer_City")["Total_Amount"].sum().sort_values(ascending=False)
city_ord = df.groupby("Customer_City")["Transaction_ID"].count()
city_pct = (city_rev / city_rev.sum() * 100).round(1)
print(pd.DataFrame({"Revenue":city_rev,"Orders":city_ord,"Pct":city_pct}).to_string())

fig, ax = plt.subplots(figsize=(12,5))
colors_c = [BRAND_GREEN if i<3 else BRAND_LIGHT if i<6 else "#A5D6A7" for i in range(len(city_rev))]
bars = ax.bar(range(len(city_rev)), city_rev.values/1000, color=colors_c, width=0.6)
ax.set_xticks(range(len(city_rev))); ax.set_xticklabels(city_rev.index, rotation=30, ha="right")
ax.set_ylabel("Revenue (PKR '000)"); ax.set_title("Revenue by Customer City", fontsize=14, fontweight="bold", pad=15)
for bar, orders in zip(bars, city_ord[city_rev.index].values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f"{bar.get_height():.0f}K\n({orders})", ha="center", va="bottom", fontsize=8, color="#333")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}04_revenue_by_city.png", dpi=160, bbox_inches="tight")
plt.close()
print("→ Chart saved: 04_revenue_by_city.png\n")

# ══════════════════════════════════════════════════════════
# ANALYSIS 5: Payment Method Distribution
# ══════════════════════════════════════════════════════════
print("━" * 50)
print("ANALYSIS 5: Payment Method Distribution")
pay_dist = df["Payment_Method"].value_counts()
pay_rev  = df.groupby("Payment_Method")["Total_Amount"].sum().sort_values()
print(f"Order count by method:\n{pay_dist.to_string()}")
print(f"\nRevenue by method:\n{pay_rev.to_string()}")
digital_pct = ((pay_dist["JazzCash"] + pay_dist["EasyPaisa"]) / pay_dist.sum() * 100)
print(f"\nDigital wallet adoption: {digital_pct:.1f}%")

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,5))
ax1.pie(pay_dist.values, labels=pay_dist.index, autopct="%1.1f%%", colors=PALETTE, startangle=140,
        pctdistance=0.82, wedgeprops=dict(edgecolor="white", linewidth=2))
ax1.set_title("Payment Method Distribution\n(by Order Count)", fontweight="bold")
ax2.barh(pay_rev.index, pay_rev.values/1000, color=PALETTE[:len(pay_rev)])
ax2.set_xlabel("Revenue (PKR '000)"); ax2.set_title("Revenue by Payment Method", fontweight="bold")
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}05_payment_methods.png", dpi=160, bbox_inches="tight")
plt.close()
print("→ Chart saved: 05_payment_methods.png\n")

# ══════════════════════════════════════════════════════════
# ANALYSIS 6: Peak Sales Periods
# ══════════════════════════════════════════════════════════
print("━" * 50)
print("ANALYSIS 6: Peak Sales Periods")
hourly = df.groupby("Hour")["Total_Amount"].sum()
daily  = df.groupby("Day_of_Week")["Total_Amount"].sum().reindex(DAY_ORDER)
peak_h = hourly.idxmax()
peak_d = daily.idxmax()
print(f"Peak Hour: {peak_h}:00 (PKR {hourly[peak_h]:,.0f})")
print(f"Peak Day: {peak_d} (PKR {daily[peak_d]:,.0f})")
print(f"Weekend Revenue: PKR {df[df['Is_Weekend']==1]['Total_Amount'].sum():,.0f} ({df[df['Is_Weekend']==1]['Total_Amount'].sum()/df['Total_Amount'].sum()*100:.1f}%)")

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))
ax1.bar(hourly.index, hourly.values/1000, color=BRAND_GREEN, alpha=0.85, width=0.7)
ax1.bar(peak_h, hourly[peak_h]/1000, color=ACCENT, alpha=1.0, width=0.7, label=f"Peak: {peak_h}:00")
ax1.set_xlabel("Hour of Day (24hr)"); ax1.set_ylabel("Revenue (PKR '000)")
ax1.set_title("Revenue by Hour of Day", fontweight="bold"); ax1.legend()
ax2.bar(daily.index, daily.values/1000, color=[ACCENT if d in ["Saturday","Sunday"] else BRAND_GREEN for d in daily.index], alpha=0.85, width=0.6)
ax2.set_xlabel("Day of Week"); ax2.set_ylabel("Revenue (PKR '000)")
ax2.set_title("Revenue by Day of Week (Orange=Weekend)", fontweight="bold"); ax2.tick_params(axis="x", rotation=30)
for ax in [ax1, ax2]: ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}06_peak_periods.png", dpi=160, bbox_inches="tight")
plt.close()
print("→ Chart saved: 06_peak_periods.png\n")

# ══════════════════════════════════════════════════════════
# ANALYSIS 7: Customer Purchasing Patterns
# ══════════════════════════════════════════════════════════
print("━" * 50)
print("ANALYSIS 7: Customer Purchasing Patterns")
cust_spend = df[df["Customer_ID"]!="CUST-UNKNOWN"].groupby("Customer_ID")["Total_Amount"].sum()
age_rev    = df.groupby("Customer_Age_Group")["Total_Amount"].mean().reindex(["18-24","25-34","35-44","45-54","55+"])
gender_rev = df[df["Customer_Gender"]!="Unknown"].groupby("Customer_Gender")["Total_Amount"].sum()
print(f"Avg Customer Lifetime Spend: PKR {cust_spend.mean():,.0f}")
print(f"Top 10% customers spend: PKR {cust_spend.quantile(0.9):,.0f}+")
print(f"Age group AOV:\n{age_rev.to_string()}")
print(f"Gender revenue split:\n{gender_rev.to_string()}")

fig, axes = plt.subplots(1,3,figsize=(17,5))
axes[0].hist(cust_spend.values, bins=30, color=BRAND_GREEN, edgecolor="white", alpha=0.85)
axes[0].axvline(cust_spend.mean(), color=ACCENT, linestyle="--", linewidth=2, label=f"Mean: PKR {cust_spend.mean():,.0f}")
axes[0].set_xlabel("Total Spend per Customer (PKR)"); axes[0].set_ylabel("Customers"); axes[0].set_title("Customer Lifetime Spend Distribution", fontweight="bold"); axes[0].legend(fontsize=9)
axes[1].bar(age_rev.index, age_rev.values, color=PALETTE[:5])
axes[1].set_xlabel("Age Group"); axes[1].set_ylabel("Avg Order Value (PKR)"); axes[1].set_title("Avg Order Value by Age Group", fontweight="bold")
axes[2].pie(gender_rev.values, labels=gender_rev.index, autopct="%1.1f%%", colors=[BRAND_GREEN, BRAND_LIGHT], startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
axes[2].set_title("Revenue Split by Gender", fontweight="bold")
for ax in axes[:2]: ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.suptitle("Customer Purchasing Patterns", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}07_customer_patterns.png", dpi=160, bbox_inches="tight")
plt.close()
print("→ Chart saved: 07_customer_patterns.png\n")

# ══════════════════════════════════════════════════════════
# ANALYSIS 8: AOV Trend & Discount Impact
# ══════════════════════════════════════════════════════════
print("━" * 50)
print("ANALYSIS 8: Average Order Value & Discount Analysis")
aov_monthly = df.groupby(["Month","Month_Name"])["Total_Amount"].mean().reset_index().sort_values("Month")
disc_rev    = df.groupby("Discount_Pct")["Total_Amount"].agg(["mean","count"]).reset_index()
print(f"Annual Average Order Value: PKR {df['Total_Amount'].mean():,.2f}")
print(f"Peak AOV Month: {aov_monthly.loc[aov_monthly['Total_Amount'].idxmax(),'Month_Name']}")
print(f"Discount distribution:\n{disc_rev.to_string()}")

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,5))
ax1.plot(aov_monthly["Month_Name"], aov_monthly["Total_Amount"], marker="o", color=BRAND_GREEN, linewidth=2.5, markersize=8)
ax1.fill_between(range(len(aov_monthly)), aov_monthly["Total_Amount"], alpha=0.15, color=BRAND_GREEN)
ax1.set_xticks(range(len(aov_monthly))); ax1.set_xticklabels(aov_monthly["Month_Name"], rotation=30)
ax1.set_ylabel("Avg Order Value (PKR)"); ax1.set_title("Monthly Average Order Value Trend", fontweight="bold")
ax1.axhline(aov_monthly["Total_Amount"].mean(), color=ACCENT, linestyle="--", alpha=0.8, label=f"Annual Avg: PKR {aov_monthly['Total_Amount'].mean():,.0f}")
ax1.legend()
disc_labels = [f"{d}%" for d in disc_rev["Discount_Pct"]]
ax2.bar(disc_labels, disc_rev["count"], color=PALETTE[:len(disc_rev)], width=0.5)
ax2_twin = ax2.twinx()
ax2_twin.plot(disc_labels, disc_rev["mean"], color=ACCENT, marker="s", linewidth=2, label="Avg Rev/Order")
ax2.set_xlabel("Discount %"); ax2.set_ylabel("Order Count", color=BRAND_GREEN); ax2_twin.set_ylabel("Avg Revenue (PKR)", color=ACCENT)
ax2.set_title("Discount Distribution & Revenue Impact", fontweight="bold"); ax2_twin.legend(loc="upper right")
for ax in [ax1, ax2]: ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}08_aov_discount.png", dpi=160, bbox_inches="tight")
plt.close()
print("→ Chart saved: 08_aov_discount.png\n")

# ══════════════════════════════════════════════════════════
# ANALYSIS 9: Category-Month Heatmap
# ══════════════════════════════════════════════════════════
print("━" * 50)
print("ANALYSIS 9: Category × Month Revenue Heatmap")
cat_monthly = df.pivot_table(values="Total_Amount", index="Product_Category", columns="Month_Name", aggfunc="sum")
cat_monthly = cat_monthly.reindex(columns=MONTH_ORDER, fill_value=0)
print(cat_monthly.round(0).to_string())

fig, ax = plt.subplots(figsize=(14,6))
sns.heatmap(cat_monthly/1000, annot=True, fmt=".0f", cmap="YlGn", ax=ax, linewidths=0.5, cbar_kws={"label":"Revenue (PKR '000)"})
ax.set_title("Category × Month Revenue Heatmap (PKR '000)", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Month"); ax.set_ylabel("Product Category")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}09_category_heatmap.png", dpi=160, bbox_inches="tight")
plt.close()
print("→ Chart saved: 09_category_heatmap.png\n")

# ══════════════════════════════════════════════════════════
# BONUS ANALYSIS 1: RFM Customer Segmentation
# ══════════════════════════════════════════════════════════
print("━" * 50)
print("BONUS 1: RFM Customer Segmentation")
snapshot = df["Purchase_Date"].max() + pd.Timedelta(days=1)
rfm = (df[df["Customer_ID"]!="CUST-UNKNOWN"]
       .groupby("Customer_ID")
       .agg(Recency=("Purchase_Date", lambda x: (snapshot - x.max()).days),
            Frequency=("Transaction_ID","count"),
            Monetary=("Total_Amount","sum"))
       .reset_index())
rfm["R_Score"] = pd.qcut(rfm["Recency"],  q=4, labels=[4,3,2,1]).astype(int)
rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=4, labels=[1,2,3,4]).astype(int)
rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"),  q=4, labels=[1,2,3,4]).astype(int)
rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]
def segment(s):
    if s>=10: return "Champions"
    elif s>=8: return "Loyal Customers"
    elif s>=6: return "Potential Loyalists"
    elif s>=4: return "At Risk"
    else: return "Lost"
rfm["Segment"] = rfm["RFM_Score"].apply(segment)
print(rfm.groupby("Segment")[["Recency","Frequency","Monetary"]].mean().round(1).to_string())
print(f"\nSegment counts:\n{rfm['Segment'].value_counts().to_string()}")
rfm.to_csv("data/rfm_segments.csv", index=False)
seg_colors = {"Champions":BRAND_GREEN,"Loyal Customers":BRAND_LIGHT,"Potential Loyalists":"#FFC107","At Risk":"#FF7043","Lost":"#B0BEC5"}
seg_counts  = rfm["Segment"].value_counts()
seg_monetary= rfm.groupby("Segment")["Monetary"].mean().sort_values()
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))
ax1.pie(seg_counts.values, labels=seg_counts.index, autopct="%1.1f%%",
        colors=[seg_colors[s] for s in seg_counts.index], startangle=140, wedgeprops=dict(edgecolor="white", linewidth=2))
ax1.set_title("Customer Segmentation (RFM)", fontweight="bold")
ax2.barh(seg_monetary.index, seg_monetary.values/1000, color=[seg_colors[s] for s in seg_monetary.index])
ax2.set_xlabel("Avg Lifetime Spend (PKR '000)"); ax2.set_title("Avg Spend by Segment", fontweight="bold")
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
plt.suptitle("RFM Customer Segmentation Analysis", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}10_customer_segmentation.png", dpi=160, bbox_inches="tight")
plt.close()
print("→ Chart saved: 10_customer_segmentation.png\n")

# ══════════════════════════════════════════════════════════
# BONUS ANALYSIS 2: Sales Forecasting (Linear Trend)
# ══════════════════════════════════════════════════════════
print("━" * 50)
print("BONUS 2: Sales Forecasting (H1 2026)")
monthly_rev = df.groupby("Month")["Total_Amount"].sum().reset_index()
monthly_rev.columns = ["Month","Revenue"]
X = monthly_rev["Month"].values.reshape(-1,1)
y = monthly_rev["Revenue"].values
model = LinearRegression().fit(X, y)
future_X    = np.array([13,14,15,16,17,18]).reshape(-1,1)
forecasts   = model.predict(future_X)
forecast_labels = ["Jan'26","Feb'26","Mar'26","Apr'26","May'26","Jun'26"]
for label, val in zip(forecast_labels, forecasts):
    print(f"  {label}: PKR {val:,.0f}")
print(f"Total H1 2026 Forecast: PKR {forecasts.sum():,.0f}")
print(f"R² Score: {model.score(X,y):.3f}")

fig, ax = plt.subplots(figsize=(13,5))
ax.bar(MONTH_ORDER, monthly_rev["Revenue"]/1000, color=BRAND_GREEN, alpha=0.8, label="Actual 2025")
ax.bar(forecast_labels, forecasts/1000, color=ACCENT, alpha=0.7, label="Forecast 2026")
ax.plot(MONTH_ORDER + forecast_labels, list(monthly_rev["Revenue"]/1000) + list(forecasts/1000),
        "o--", color="#333", linewidth=1.5, markersize=5)
ax.axvline(x=11.5, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
ax.text(11.6, max(monthly_rev["Revenue"].max(), forecasts.max())/1000*0.9, "→ Forecast →", color="gray", fontsize=10)
ax.set_ylabel("Revenue (PKR '000)"); ax.set_title("Sales Forecast: H1 2026 (Linear Trend)", fontsize=14, fontweight="bold", pad=15)
ax.tick_params(axis="x", rotation=30); ax.legend()
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}11_sales_forecast.png", dpi=160, bbox_inches="tight")
plt.close()
print("→ Chart saved: 11_sales_forecast.png\n")

print("═" * 50)
print("  ALL EDA ANALYSES COMPLETE")
print("  Charts saved to: charts/")
print("  RFM data saved to: data/rfm_segments.csv")
print("═" * 50)
