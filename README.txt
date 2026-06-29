================================================================
  NOVAMART RETAIL ANALYTICS PROJECT
  Task DA-2 | TEYZIX CORE Internship Program | June 2026
================================================================

PROJECT OVERVIEW
----------------
Company     : NovaMart General Stores Pvt. Ltd.
Industry    : Multi-Category Retail (B2C)
Analysis    : Retail Store Sales & Customer Behavior Analysis
Dataset     : 1,095 records | 22 columns | Jan–Dec 2025
Tools Used  : Python 3.x, Pandas, NumPy, Matplotlib, Seaborn,
              Scikit-learn, Docx (Node.js)

----------------------------------------------------------------
FOLDER STRUCTURE
----------------------------------------------------------------
Task-2/
├── data/
│   ├── novamart_raw.csv              ← Raw dataset (1,150 rows, with issues)
│   ├── novamart_cleaned.csv          ← Final cleaned dataset (1,095 rows)
│   └── rfm_segments.csv              ← RFM customer segmentation output
│
├── charts/                           ← All 11 EDA visualizations (PNG)
│   ├── 01_monthly_revenue.png
│   ├── 02_top_products.png
│   ├── 03_category_performance.png
│   ├── 04_revenue_by_city.png
│   ├── 05_payment_methods.png
│   ├── 06_peak_periods.png
│   ├── 07_customer_patterns.png
│   ├── 08_aov_discount.png
│   ├── 09_category_heatmap.png
│   ├── 10_customer_segmentation.png
│   └── 11_sales_forecast.png
│
├── report/
│   └── NovaMart_Analytics_Report_DA2.docx  ← Full professional report
│
├── 01_generate_dataset_v2.py         ← Dataset generation script
├── 02_data_cleaning.py               ← Data cleaning pipeline
├── 03_eda_complete.py                ← Complete EDA + Bonus Analysis
└── README.txt                        ← This file

----------------------------------------------------------------
HOW TO RUN
----------------------------------------------------------------
Step 1: Install dependencies
    pip install pandas numpy matplotlib seaborn scikit-learn

Step 2: Generate raw dataset
    python 01_generate_dataset_v2.py

Step 3: Run data cleaning
    python 02_data_cleaning.py

Step 4: Run full EDA & generate all charts
    python 03_eda_complete.py

All outputs will appear in data/ and charts/ directories.

----------------------------------------------------------------
KEY RESULTS SUMMARY
----------------------------------------------------------------
Total Revenue (2025)  : PKR 9,488,640
Total Orders          : 1,095
Unique Customers      : 859
Average Order Value   : PKR 8,665
Top Category          : Electronics (24.3% of revenue)
Top City              : Lahore (20.7% of revenue)
Top Product           : Samsung Galaxy A14
Peak Shopping Hour    : 15:00 (3 PM)
Digital Payment Share : 26% (JazzCash + EasyPaisa)
H1 2026 Forecast      : PKR 5,307,300 (+6.8% YoY)

RFM Segments:
  Champions           : 167 customers (19.4%)
  Loyal Customers     : 253 customers (29.5%)
  Potential Loyalists : 278 customers (32.4%)
  At Risk             : 139 customers (16.2%)
  Lost                : 22  customers (2.6%)

================================================================
