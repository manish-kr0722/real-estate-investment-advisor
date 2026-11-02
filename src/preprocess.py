"""Step 1 - Data preprocessing & feature engineering.
Usage: python src/preprocess.py [raw_csv] [out_csv]
"""
import sys, json
import numpy as np, pandas as pd

RAW = sys.argv[1] if len(sys.argv) > 1 else "data/india_housing_prices.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "outputs/india_housing_cleaned.csv"
CURRENT_YEAR = 2025          # dataset's Age_of_Property = 2025 - Year_Built
HORIZON = 5                  # forecast horizon (years)
GOOD_GROWTH_THRESHOLD = 0.07  # annual appreciation needed to be a "Good Investment"
AMENITY_LIST = ["Playground", "Gym", "Garden", "Pool", "Clubhouse"]
TRANSPORT_MAP = {"Low": 1, "Medium": 2, "High": 3}

df = pd.read_csv(RAW)
log = {"rows_raw": len(df)}

# ---- 1. missing values & duplicates -------------------------------------
log["missing_cells_raw"] = int(df.isna().sum().sum())
log["duplicate_rows_raw"] = int(df.duplicated().sum())
log["duplicate_ids_raw"] = int(df["ID"].duplicated().sum())
df = df.drop_duplicates().drop_duplicates(subset="ID")
# generic safety net (no-ops on this file, but keeps the script reusable)
for c in df.select_dtypes("number"):
    df[c] = df[c].fillna(df[c].median())
for c in df.select_dtypes(exclude="number"):
    df[c] = df[c].fillna(df[c].mode().iloc[0])
for c in df.select_dtypes(exclude="number"):
    df[c] = df[c].astype(str).str.strip()

# ---- 2. consistency checks ----------------------------------------------
df["Age_of_Property"] = CURRENT_YEAR - df["Year_Built"]
df["Floor_Inconsistent"] = (df["Floor_No"] > df["Total_Floors"]).astype(int)
log["floor_gt_total_floors_pct"] = round(df["Floor_Inconsistent"].mean() * 100, 2)

# ---- 3. price per sq ft (Rs) --------------------------------------------
# The source Price_per_SqFt is in lakhs and rounded to 2 decimals (loses precision).
df["Price_per_SqFt_Rs"] = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]

# ---- 4. outlier flags (IQR rule) ----------------------------------------
def iqr_flag(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return ((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).astype(int)
for c in ["Price_per_SqFt_Rs", "Size_in_SqFt", "Price_in_Lakhs"]:
    df[f"Outlier_{c}"] = iqr_flag(df[c])
    log[f"outliers_{c}"] = int(df[f"Outlier_{c}"].sum())

# ---- 5. feature engineering ---------------------------------------------
am = df["Amenities"].fillna("")
for a in AMENITY_LIST:
    df[f"Has_{a}"] = am.str.contains(a).astype(int)
df["Amenity_Count"] = df[[f"Has_{a}" for a in AMENITY_LIST]].sum(axis=1)
df["Parking_Flag"] = (df["Parking_Space"] == "Yes").astype(int)
df["Security_Flag"] = (df["Security"] == "Yes").astype(int)
df["Transport_Score"] = df["Public_Transport_Accessibility"].map(TRANSPORT_MAP)
df["Under_Construction"] = (df["Availability_Status"] == "Under_Construction").astype(int)
df["School_Density_Score"] = df["Nearby_Schools"] / df["Nearby_Schools"].max() * 10
df["Hospital_Density_Score"] = df["Nearby_Hospitals"] / df["Nearby_Hospitals"].max() * 10
df["Price_per_BHK"] = df["Price_in_Lakhs"] / df["BHK"]
df["Size_per_BHK"] = df["Size_in_SqFt"] / df["BHK"]
df["Floor_Ratio"] = (df["Floor_No"].clip(upper=df["Total_Floors"]) / df["Total_Floors"]).round(3)

# Infrastructure score 0-100: schools 20, hospitals 20, transport 25, amenities 20, parking 7.5, security 7.5
df["Infrastructure_Score"] = (
    20 * df["Nearby_Schools"] / 10 + 20 * df["Nearby_Hospitals"] / 10
    + 25 * (df["Transport_Score"] - 1) / 2 + 20 * df["Amenity_Count"] / 5
    + 7.5 * df["Parking_Flag"] + 7.5 * df["Security_Flag"])

# relative value: price/sqft vs the median of the same city & property type
df["City_Median_PPSF"] = df.groupby(["City", "Property_Type"])["Price_per_SqFt_Rs"].transform("median")
df["Value_Ratio"] = df["Price_per_SqFt_Rs"] / df["City_Median_PPSF"]

# ---- 6. targets (domain rules - dataset has no historical prices) ---------
# Assumed annual appreciation: 5% base + up to 3% infrastructure, +1% under construction
# (developing-area premium), +0.5% new build (<=10y), -0.5% old build (>25y).
df["Est_Annual_Growth"] = (0.05 + 0.03 * df["Infrastructure_Score"] / 100
                           + 0.01 * df["Under_Construction"]
                           + 0.005 * (df["Age_of_Property"] <= 10)
                           - 0.005 * (df["Age_of_Property"] > 25))
df["Future_Price_5Y_Lakhs"] = (df["Price_in_Lakhs"] * (1 + df["Est_Annual_Growth"]) ** HORIZON).round(2)
# Good investment = fairly/under-priced for its city & type AND strong expected growth
df["Good_Investment"] = ((df["Value_Ratio"] <= 1.0)
                         & (df["Est_Annual_Growth"] >= GOOD_GROWTH_THRESHOLD)).astype(int)

log.update(rows_clean=len(df), cols_clean=df.shape[1],
           good_investment_pct=round(df["Good_Investment"].mean() * 100, 2),
           growth_min=round(df.Est_Annual_Growth.min(), 4), growth_max=round(df.Est_Annual_Growth.max(), 4))
df.to_csv(OUT, index=False)
json.dump(log, open("outputs/preprocess_log.json", "w"), indent=2)
print(json.dumps(log, indent=2))
