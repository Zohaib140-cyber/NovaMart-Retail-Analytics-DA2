"""
NovaMart Retail Analytics – Dataset Generation Script v2
"""
import pandas as pd
import numpy as np
import random
from datetime import datetime

import os
os.makedirs("data", exist_ok=True)
os.makedirs("charts", exist_ok=True)


np.random.seed(42)
random.seed(42)

CITIES = ["Karachi","Lahore","Islamabad","Rawalpindi","Faisalabad","Multan","Peshawar","Quetta","Hyderabad","Sialkot"]
CITY_WEIGHTS = [0.22,0.20,0.12,0.10,0.09,0.08,0.07,0.05,0.04,0.03]
PAYMENT_METHODS = ["Cash","Credit Card","Debit Card","JazzCash","EasyPaisa","Bank Transfer"]
PAY_WEIGHTS = [0.30,0.22,0.18,0.14,0.12,0.04]
GENDERS = ["Male","Female"]
AGE_GROUPS = ["18-24","25-34","35-44","45-54","55+"]
STORE_NAMES = ["NovaMart Central","NovaMart Express","NovaMart Premium","NovaMart Outlet"]

CATALOGUE = {
    "Electronics":[("Samsung Galaxy A14",(45000,52000)),("Haier LED TV 32\"",(38000,44000)),("JBL Bluetooth Speaker",(4500,7500)),("HP Wireless Mouse",(1200,2000)),("Anker USB-C Charger",(800,1500)),("Sony Earbuds WF-C500",(8500,12000)),("Xiaomi Power Bank 20000mAh",(3200,4800)),("Lenovo Laptop Bag",(2500,4000))],
    "Clothing":[("Men's Polo Shirt",(1200,2500)),("Women's Kameez Printed",(1500,3200)),("Denim Jeans Slim Fit",(2200,4000)),("Kids Tracksuit",(1000,1800)),("Winter Jacket",(4500,8000)),("Sports T-Shirt",(900,1600)),("Casual Sneakers",(3500,6500)),("Formal Dress Shirt",(1800,3000))],
    "Groceries":[("Basmati Rice 5kg",(1200,1600)),("Sunflower Oil 5L",(1800,2200)),("Olpers Full Cream Milk 1L",(180,220)),("Nestle Everyday Tea Whitener 400g",(350,420)),("Tapal Danedar Tea 900g",(700,850)),("Rooh Afza 750ml",(280,350)),("Brooke Bond Supreme 200g",(280,340)),("Lays Chips Family Pack",(150,200))],
    "Home & Kitchen":[("Anex Blender AG-6122",(3200,4500)),("Dawlance Microwave 25L",(18000,24000)),("Prestige Pressure Cooker 5L",(2800,4000)),("Pillow Set 2pcs",(1200,2000)),("Bed Sheet King Size",(2500,4500)),("Kitchen Knife Set",(800,1500)),("Non-Stick Fry Pan",(1500,2800)),("PEL Refrigerator 14CuFt",(70000,85000))],
    "Sports & Fitness":[("Adidas Lite Racer Shoes",(6500,9000)),("Yoga Mat 6mm",(1500,2500)),("Dumbbell Set 10kg Pair",(3500,5500)),("Wilson Badminton Racket",(2200,3800)),("Football Size 5",(1800,2800)),("Cycling Gloves",(600,1200)),("Resistance Bands Set",(900,1600)),("Skipping Rope Steel",(400,700))],
    "Health & Beauty":[("Ponds Moisturizer 200ml",(700,950)),("Dettol Handwash 500ml",(280,380)),("Colgate Toothpaste 150g",(180,250)),("Pantene Shampoo 360ml",(480,620)),("Gillette Fusion Razor",(1200,1800)),("LOreal Face Wash",(550,800)),("Panadol Strip 10tabs",(80,120)),("Himalaya Neem Face Wash",(350,480))],
    "Stationery":[("Camlin Geometry Box",(280,420)),("A4 Paper Ream 500sheets",(700,950)),("Ball Pen Pack 10pcs",(150,220)),("Stapler Heavy Duty",(550,900)),("Casio fx-82 Calculator",(1800,2500)),("Spiral Notebook A5",(250,380)),("Highlighter Set 6colors",(350,550)),("Sticky Notes 3x3 4pads",(200,320))],
    "Toys & Kids":[("LEGO Classic Bricks Set",(4500,7000)),("Barbie Fashion Doll",(1800,2800)),("Remote Control Car",(3200,5500)),("Jigsaw Puzzle 500pcs",(900,1500)),("Ludo Deluxe Board Game",(550,900)),("Water Gun Set",(400,700)),("Art & Craft Kit",(1200,2000)),("Stuffed Teddy Bear 30cm",(800,1400))],
}
CATEGORIES = list(CATALOGUE.keys())
CAT_WEIGHTS = [12,15,20,10,8,10,8,7]

SEASONAL = {
    "Electronics":    [0.9,0.8,0.9,1.0,1.0,1.1,1.2,1.1,1.0,1.2,1.5,1.6],
    "Clothing":       [1.2,0.8,1.3,1.0,0.9,0.8,1.1,1.0,1.2,1.0,1.4,1.5],
    "Groceries":      [1.0,1.0,1.0,1.1,1.2,1.3,1.1,1.0,1.1,1.0,1.0,1.2],
    "Home & Kitchen": [0.8,0.9,1.2,1.1,1.0,0.9,0.8,0.9,1.0,1.1,1.3,1.4],
    "Sports & Fitness":[1.0,1.1,1.2,1.2,1.0,0.8,0.7,0.9,1.1,1.2,1.0,0.9],
    "Health & Beauty":[1.1,1.0,1.0,1.0,1.1,1.2,1.1,1.0,1.0,1.1,1.1,1.2],
    "Stationery":     [1.2,1.1,0.9,0.8,0.8,1.4,1.5,1.3,1.0,0.9,0.8,0.9],
    "Toys & Kids":    [0.7,0.8,0.8,0.9,1.0,1.0,1.0,1.1,0.9,1.0,1.2,1.8],
}

DAYS_IN_MONTH = [31,28,31,30,31,30,31,31,30,31,30,31]

customer_pool = [f"CUST-{i:05d}" for i in range(1,3001)]
records = []

for i in range(1100):
    month = random.randint(1,12)
    day   = random.randint(1, DAYS_IN_MONTH[month-1])
    hour  = random.choices(range(9,22), weights=[1,2,3,4,5,6,7,7,6,5,4,3,2], k=1)[0]
    minute= random.randint(0,59)
    dt    = datetime(2025, month, day, hour, minute)

    category = random.choices(CATEGORIES, weights=CAT_WEIGHTS, k=1)[0]
    s_mult   = SEASONAL[category][month-1]
    if random.random() > s_mult/1.8:
        category = random.choices(CATEGORIES, weights=CAT_WEIGHTS, k=1)[0]

    prod, pr = random.choice(CATALOGUE[category])
    unit_price = round(random.uniform(*pr), 2)
    qty        = random.choices([1,2,3,4,5], weights=[50,25,12,8,5], k=1)[0]
    discount   = random.choices([0,5,10,15,20], weights=[55,20,13,8,4], k=1)[0]
    total_amt  = round(unit_price * qty * (1 - discount/100), 2)
    city       = random.choices(CITIES, weights=CITY_WEIGHTS, k=1)[0]
    payment    = random.choices(PAYMENT_METHODS, weights=PAY_WEIGHTS, k=1)[0]
    gender     = random.choice(GENDERS)
    age_grp    = random.choices(AGE_GROUPS, weights=[15,30,25,18,12], k=1)[0]
    rating     = random.choices([1,2,3,4,5], weights=[3,7,15,40,35], k=1)[0]
    store      = random.choice(STORE_NAMES)

    records.append({
        "Transaction_ID": f"TXN-{2025000+i+1:07d}",
        "Customer_ID": random.choice(customer_pool),
        "Product_Category": category,
        "Product_Name": prod,
        "Quantity": qty,
        "Unit_Price": unit_price,
        "Discount_Pct": discount,
        "Total_Amount": total_amt,
        "Purchase_Date": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Customer_City": city,
        "Payment_Method": payment,
        "Store_Name": store,
        "Customer_Gender": gender,
        "Customer_Age_Group": age_grp,
        "Customer_Rating": rating,
    })

df_clean = pd.DataFrame(records)
df_raw   = df_clean.copy()

# Inject issues
for col in ["Customer_ID","Payment_Method","Customer_Rating","Customer_Gender","Discount_Pct"]:
    idx = df_raw.sample(frac=0.05, random_state=42).index
    df_raw.loc[idx, col] = np.nan

idx2 = df_raw.sample(frac=0.02, random_state=7).index
df_raw.loc[idx2, ["Unit_Price","Total_Amount"]] = np.nan

dupes = df_raw.sample(n=50, random_state=99)
df_raw = pd.concat([df_raw, dupes], ignore_index=True)

city_map  = {"Karachi":["karachi","KARACHI","Karachii"],"Lahore":["lahore","LAHORE","Lahor"],"Islamabad":["islamabad","ISLAMABAD"],"Rawalpindi":["rawalpindi","RAWALPINDI","Rawal Pindi"]}
pay_map   = {"Cash":["cash","CASH"],"Credit Card":["credit card","CreditCard","Credit card"],"JazzCash":["jazzcash","Jazz Cash"]}
for correct, variants in city_map.items():
    idx = df_raw[df_raw["Customer_City"]==correct].sample(frac=0.10, random_state=1).index
    df_raw.loc[idx,"Customer_City"] = [random.choice(variants) for _ in idx]
for correct, variants in pay_map.items():
    idx = df_raw[df_raw["Payment_Method"]==correct].dropna().sample(frac=0.10, random_state=2).index
    df_raw.loc[idx,"Payment_Method"] = [random.choice(variants) for _ in idx]

bad = df_raw.sample(n=8, random_state=55).index
df_raw.loc[bad,"Quantity"] = random.choices([-1,0], k=8)

bad2 = df_raw.sample(n=5, random_state=66).index
df_raw.loc[bad2,"Purchase_Date"] = "2030-01-01 00:00:00"

df_raw = df_raw.sample(frac=1, random_state=42).reset_index(drop=True)
df_raw.to_csv("data/novamart_raw.csv", index=False)
df_clean.to_csv("data/novamart_reference_clean.csv", index=False)

print(f"Raw dataset: {df_raw.shape}")
print(f"Clean reference: {df_clean.shape}")
print("\nMissing values:\n", df_raw.isnull().sum())
print(f"\nDuplicates: {df_raw.duplicated().sum()}")
