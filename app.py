"""Real Estate Investment Advisor - Streamlit app.   Run:  streamlit run app.py
Needs (created by the pipeline): models/best_classifier.joblib, models/best_regressor.joblib,
outputs/model_results.json, outputs/app_sample.csv"""
import json, joblib, pandas as pd, numpy as np, streamlit as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
import sys, os
from pathlib import Path
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))
from features import add_features, FEATURES, AMENITIES

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")

REQUIRED = ["models/best_classifier.joblib", "models/best_regressor.joblib", "outputs/app_sample.csv", "outputs/model_results.json", "src/features.py"]
missing = [f for f in REQUIRED if not (BASE / f).exists()]
if missing:
    st.error("These files are missing from the repository (upload them with the same folder names):")
    st.code("\n".join(missing)); st.stop()

@st.cache_resource
def load_models():
    return joblib.load(BASE / "models/best_classifier.joblib"), joblib.load(BASE / "models/best_regressor.joblib")
@st.cache_data
def load_data(): return pd.read_csv(BASE / "outputs/app_sample.csv")
@st.cache_data
def load_results(): return json.load(open(BASE / "outputs/model_results.json"))

clf, reg = load_models(); data = load_data(); res = load_results()
city_median = data.groupby(["City", "Property_Type"]).Price_per_SqFt_Rs.median()

def predict(inp: dict):
    d = pd.DataFrame([inp]); d["Amenities"] = ", ".join(inp["amenities_list"]); d = d.drop(columns=["amenities_list"])
    X = add_features(d)[FEATURES]
    p = float(clf.predict_proba(X)[0, 1]); price5 = float(reg.predict(X)[0])
    return p, price5, X

st.title("🏠 Real Estate Investment Advisor")
st.caption("Predict whether a property is a **Good Investment** and its **estimated price after 5 years**. "
           "Labels are rule-based (see Insights tab) - this is an educational decision-support tool, not financial advice.")
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "🔎 Browse & filter", "📊 Market insights", "🧠 Model info"])

# ------------------------------------------------------------------ Predict
with tab1:
    st.subheader("Enter property details")
    states = sorted(data.State.unique())
    c = st.columns(4)
    state = c[0].selectbox("State", states)
    city = c[1].selectbox("City", sorted(data[data.State == state].City.unique()))
    ptype = c[2].selectbox("Property type", sorted(data.Property_Type.unique()))
    bhk = c[3].slider("BHK", 1, 5, 3)
    c = st.columns(4)
    size = c[0].number_input("Size (sq ft)", 500, 5000, 1800, 50)
    price = c[1].number_input("Asking price (Lakhs)", 10.0, 500.0, 120.0, 1.0)
    year = c[2].number_input("Year built", 1990, 2025, 2012)
    furn = c[3].selectbox("Furnishing", ["Unfurnished", "Semi-furnished", "Furnished"])
    c = st.columns(4)
    tot = c[0].number_input("Total floors", 1, 30, 12); floor = c[1].number_input("Floor no.", 0, 30, 4)
    schools = c[2].slider("Nearby schools", 1, 10, 5); hosp = c[3].slider("Nearby hospitals", 1, 10, 5)
    c = st.columns(4)
    transport = c[0].selectbox("Public transport access", ["Low", "Medium", "High"], 1)
    parking = c[1].selectbox("Parking space", ["Yes", "No"]); security = c[2].selectbox("Security", ["Yes", "No"])
    facing = c[3].selectbox("Facing", ["North", "South", "East", "West"])
    c = st.columns(3)
    owner = c[0].selectbox("Owner type", ["Owner", "Builder", "Broker"])
    avail = c[1].selectbox("Availability", ["Ready_to_Move", "Under_Construction"])
    ams = c[2].multiselect("Amenities", AMENITIES, ["Gym", "Pool"])
    if st.button("Analyse investment", type="primary"):
        inp = dict(State=state, City=city, Property_Type=ptype, BHK=bhk, Size_in_SqFt=size, Price_in_Lakhs=price, Year_Built=year,
                   Furnished_Status=furn, Floor_No=floor, Total_Floors=tot, Nearby_Schools=schools, Nearby_Hospitals=hosp,
                   Public_Transport_Accessibility=transport, Parking_Space=parking, Security=security, Facing=facing,
                   Owner_Type=owner, Availability_Status=avail, amenities_list=ams)
        p, p5, X = predict(inp)
        cagr = (p5 / price) ** (1 / 5) - 1
        ppsf = price * 1e5 / size; med = city_median.get((city, ptype), np.nan)
        a, b, c3 = st.columns(3)
        a.metric("Is this a Good Investment?", "✅ YES" if p >= .5 else "❌ NO", f"confidence {max(p, 1-p):.0%}")
        b.metric("Estimated price after 5 years", f"₹{p5:,.1f} L", f"{p5-price:+,.1f} L  ({cagr:.1%} / yr)")
        c3.metric("Price per sq ft vs city median", f"₹{ppsf:,.0f}", f"{(ppsf/med-1):+.0%} vs {city} {ptype}" if med == med else None, delta_color="inverse")
        st.progress(p, text=f"Probability of 'Good Investment': {p:.1%}")
        yrs = np.arange(0, 6); fig, ax = plt.subplots(figsize=(6, 3)); ax.plot(yrs, price * (p5 / price) ** (yrs / 5), marker="o")
        ax.set(xlabel="Years from now", ylabel="Price (Lakhs)", title="Projected value (constant-growth path)"); st.pyplot(fig)

# ------------------------------------------------------------------ Browse
with tab2:
    st.subheader("Filter listings")
    f = st.columns(4)
    fs = f[0].multiselect("State", states); ft = f[1].multiselect("Property type", sorted(data.Property_Type.unique()))
    fb = f[2].multiselect("BHK", [1, 2, 3, 4, 5]); fg = f[3].selectbox("Show", ["All", "Good investments only"])
    r = st.columns(2)
    pr = r[0].slider("Price (Lakhs)", 10, 500, (10, 500)); sz = r[1].slider("Area (sq ft)", 500, 5000, (500, 5000))
    q = data[data.Price_in_Lakhs.between(*pr) & data.Size_in_SqFt.between(*sz)]
    if fs: q = q[q.State.isin(fs)]
    if ft: q = q[q.Property_Type.isin(ft)]
    if fb: q = q[q.BHK.isin(fb)]
    if fg != "All": q = q[q.Good_Investment == 1]
    st.write(f"**{len(q):,}** properties match (from a 40,000-row sample)  |  good-investment share: **{q.Good_Investment.mean():.0%}**" if len(q) else "No matches")
    st.dataframe(q[["State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_Rs", "Year_Built", "Good_Investment", "Future_Price_5Y_Lakhs"]].head(500), use_container_width=True)

# ------------------------------------------------------------------ Insights
with tab3:
    st.subheader("Market insights")
    c1, c2 = st.columns(2)
    hm = data.pivot_table(index="City", columns="Property_Type", values="Price_per_SqFt_Rs", aggfunc="median")
    fig, ax = plt.subplots(figsize=(5, 9)); sns.heatmap(hm, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Median Rs/sq ft"}); ax.set_title("Heatmap: median price/sq ft by city & type"); c1.pyplot(fig)
    tr = data.assign(Cohort=data.Year_Built // 5 * 5).groupby("Cohort").Price_in_Lakhs.mean()
    fig, ax = plt.subplots(figsize=(5, 3.5)); ax.plot(tr.index, tr.values, marker="o"); ax.set(ylim=(0, tr.max() * 1.3), title="Avg price by build-year cohort", ylabel="Lakhs"); c2.pyplot(fig)
    gs = data.groupby("State").Good_Investment.mean().sort_values() * 100
    fig, ax = plt.subplots(figsize=(5, 5)); gs.plot.barh(ax=ax); ax.set(title="% Good Investment by state", xlabel="%"); c2.pyplot(fig)
    st.info("**How labels are defined.** Good Investment = price/sq ft ≤ the city & property-type median AND estimated annual growth ≥ 7%. "
            "Growth = 5% base + up to 3% from infrastructure score + 1% if under construction + 0.5% if ≤10 yrs old − 0.5% if >25 yrs. "
            "The 5-year price = price × (1+growth)^5. The source data contains no historical prices, so these are assumptions, not observed returns.")

# ------------------------------------------------------------------ Model info
with tab4:
    st.subheader("Model performance & feature importance")
    runs = pd.DataFrame([{"task": r["task"], "model": r["model"], **r["metrics"]} for r in res["runs"]])
    st.write("**Classification** (hold-out 20%)"); st.dataframe(runs[runs.task == "classification"].dropna(axis=1, how="all").drop(columns="task"), hide_index=True)
    st.write("**Regression** (price after 5 years, Lakhs)"); st.dataframe(runs[runs.task == "regression"].dropna(axis=1, how="all").drop(columns="task"), hide_index=True)
    c1, c2 = st.columns(2)
    for col, k, t in [(c1, "clf", "Classifier"), (c2, "reg", "Regressor")]:
        s = pd.Series(res["importance"][k]).sort_values(); fig, ax = plt.subplots(figsize=(5, 4)); s.plot.barh(ax=ax); ax.set_title(f"{t} - permutation importance"); col.pyplot(fig)
    st.caption("Metrics are very high because the targets are generated from rules over the same inputs; the models recover those rules. "
               "Treat them as a demonstration of the pipeline, not proof of real-world predictive power.")
