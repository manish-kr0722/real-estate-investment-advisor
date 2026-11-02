"""Shared feature builder - used by train.py and the Streamlit app so both stay identical."""
import numpy as np, pandas as pd

CURRENT_YEAR = 2025
NUM = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_Rs", "Year_Built", "Floor_No", "Total_Floors", "Age_of_Property",
       "Nearby_Schools", "Nearby_Hospitals", "Transport_Score", "Parking_Flag", "Security_Flag", "Amenity_Count",
       "Under_Construction", "School_Density_Score", "Hospital_Density_Score", "Price_per_BHK", "Size_per_BHK", "Floor_Ratio"]
CAT = ["State", "City", "Property_Type", "Furnished_Status", "Facing", "Owner_Type"]
FEATURES = NUM + CAT
AMENITIES = ["Playground", "Gym", "Garden", "Pool", "Clubhouse"]

def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["Age_of_Property"] = CURRENT_YEAR - d["Year_Built"]
    d["Price_per_SqFt_Rs"] = d["Price_in_Lakhs"] * 1e5 / d["Size_in_SqFt"]
    d["Parking_Flag"] = (d["Parking_Space"] == "Yes").astype(int)
    d["Security_Flag"] = (d["Security"] == "Yes").astype(int)
    d["Transport_Score"] = d["Public_Transport_Accessibility"].map({"Low": 1, "Medium": 2, "High": 3})
    d["Under_Construction"] = (d["Availability_Status"] == "Under_Construction").astype(int)
    d["Amenity_Count"] = sum(d["Amenities"].fillna("").str.contains(a).astype(int) for a in AMENITIES)
    d["School_Density_Score"] = d["Nearby_Schools"] / 10 * 10
    d["Hospital_Density_Score"] = d["Nearby_Hospitals"] / 10 * 10
    d["Price_per_BHK"] = d["Price_in_Lakhs"] / d["BHK"]
    d["Size_per_BHK"] = d["Size_in_SqFt"] / d["BHK"]
    d["Floor_Ratio"] = (d["Floor_No"].clip(upper=d["Total_Floors"]) / d["Total_Floors"]).round(3)
    return d
