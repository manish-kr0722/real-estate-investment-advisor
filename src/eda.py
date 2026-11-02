"""Step 2 - EDA: answers the 20 questions, saves figures + outputs/eda_findings.json"""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats

df = pd.read_csv("outputs/india_housing_cleaned.csv")
F = "outputs/figures/"; R = {}
sns.set_theme(style="whitegrid", font_scale=.95)
P = sns.color_palette("deep")
def save(n): plt.tight_layout(); plt.savefig(F + n, dpi=130); plt.close()
def anova(col, y="Price_per_SqFt_Rs"):
    g = [x[y].values for _, x in df.groupby(col)]
    f, p = stats.f_oneway(*g); return {"F": round(float(f), 2), "p": float(p)}
def spear(a, b):
    r, p = stats.spearmanr(df[a], df[b]); return {"r": round(float(r), 4), "p": float(p)}

# ---- Q1 price distribution
s = df.Price_in_Lakhs
R["q1"] = dict(mean=s.mean(), median=s.median(), std=s.std(), min=s.min(), max=s.max(),
               skew=float(s.skew()), q25=s.quantile(.25), q75=s.quantile(.75))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(s, bins=50, ax=ax[0], color=P[0]); ax[0].set(title="Q1 Price distribution", xlabel="Price (Lakhs)")
sns.boxplot(x=s, ax=ax[1], color=P[0]); ax[1].set(title="Price box plot", xlabel="Price (Lakhs)"); save("q01_price_distribution.png")
# ---- Q2 size
s = df.Size_in_SqFt
R["q2"] = dict(mean=s.mean(), median=s.median(), std=s.std(), min=s.min(), max=s.max(), skew=float(s.skew()))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(s, bins=50, ax=ax[0], color=P[1]); ax[0].set(title="Q2 Size distribution", xlabel="Sq ft")
sns.boxplot(x=s, ax=ax[1], color=P[1]); ax[1].set(title="Size box plot", xlabel="Sq ft"); save("q02_size_distribution.png")
# ---- Q3 PPSF by property type
t = df.groupby("Property_Type").Price_per_SqFt_Rs.agg(["mean", "median", "count"]).round(0)
R["q3"] = dict(table=t.reset_index().to_dict("records"), anova=anova("Property_Type"))
plt.figure(figsize=(7, 4.5)); sns.boxplot(data=df, x="Property_Type", y="Price_per_SqFt_Rs", hue="Property_Type", showfliers=False, legend=False)
plt.title("Q3 Price per sq ft by property type (outliers hidden)"); plt.ylabel("Rs per sq ft"); save("q03_ppsf_property_type.png")
# ---- Q4 size vs price
R["q4"] = dict(pearson=float(df.Size_in_SqFt.corr(df.Price_in_Lakhs)), spearman=spear("Size_in_SqFt", "Price_in_Lakhs"),
               pearson_ppsf=float(df.Size_in_SqFt.corr(df.Price_per_SqFt_Rs)), spearman_ppsf=spear("Size_in_SqFt", "Price_per_SqFt_Rs"))
sm = df.sample(8000, random_state=1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
ax[0].scatter(sm.Size_in_SqFt, sm.Price_in_Lakhs, s=4, alpha=.35); ax[0].set(title="Q4 Size vs price (8k sample)", xlabel="Sq ft", ylabel="Lakhs")
ax[1].scatter(sm.Size_in_SqFt, sm.Price_per_SqFt_Rs, s=4, alpha=.35, color=P[3]); ax[1].set(title="Size vs price per sq ft", xlabel="Sq ft", ylabel="Rs/sq ft", yscale="log"); save("q04_size_vs_price.png")
# ---- Q5 outliers
def iqr(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1; lo, hi = q1 - 1.5 * i, q3 + 1.5 * i
    return dict(lower=lo, upper=hi, n_outliers=int(((s < lo) | (s > hi)).sum()), pct=float(((s < lo) | (s > hi)).mean() * 100))
R["q5"] = dict(ppsf=iqr(df.Price_per_SqFt_Rs), size=iqr(df.Size_in_SqFt), price=iqr(df.Price_in_Lakhs),
               ppsf_p99=float(df.Price_per_SqFt_Rs.quantile(.99)), ppsf_max=float(df.Price_per_SqFt_Rs.max()), ppsf_min=float(df.Price_per_SqFt_Rs.min()),
               ppsf_median=float(df.Price_per_SqFt_Rs.median()), ppsf_skew=float(df.Price_per_SqFt_Rs.skew()))
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a, c, n in zip(ax, ["Price_per_SqFt_Rs", "Size_in_SqFt", "Price_in_Lakhs"], ["Price/sq ft (Rs)", "Size (sq ft)", "Price (Lakhs)"]):
    sns.boxplot(y=df[c], ax=a, color=P[2]); a.set_title("Q5 " + n)
save("q05_outliers.png")
# ---- Q6 PPSF by state
st = df.groupby("State").Price_per_SqFt_Rs.agg(["mean", "median"]).sort_values("mean", ascending=False).round(0)
R["q6"] = dict(top5=st.head(5).reset_index().to_dict("records"), bottom5=st.tail(5).reset_index().to_dict("records"),
               spread_pct=float((st["mean"].max() / st["mean"].min() - 1) * 100), anova=anova("State"), overall_mean=float(df.Price_per_SqFt_Rs.mean()))
plt.figure(figsize=(8, 6)); sns.barplot(x=st["mean"], y=st.index, color=P[0]); plt.axvline(df.Price_per_SqFt_Rs.mean(), ls="--", c="k", label="national mean")
plt.title("Q6 Average price per sq ft by state"); plt.xlabel("Rs per sq ft"); plt.legend(); save("q06_ppsf_state.png")
# ---- Q7 avg price by city
ct = df.groupby("City").Price_in_Lakhs.mean().sort_values(ascending=False).round(1)
R["q7"] = dict(top5=ct.head(5).to_dict(), bottom5=ct.tail(5).to_dict(), spread_pct=float((ct.max() / ct.min() - 1) * 100), anova=anova("City", "Price_in_Lakhs"), overall=float(df.Price_in_Lakhs.mean()))
plt.figure(figsize=(8, 9)); sns.barplot(x=ct.values, y=ct.index, color=P[1]); plt.axvline(df.Price_in_Lakhs.mean(), ls="--", c="k")
plt.title("Q7 Average property price by city (Lakhs)"); plt.xlabel("Lakhs"); save("q07_price_city.png")
# ---- Q8 median age by locality
la = df.groupby("Locality").Age_of_Property.median()
R["q8"] = dict(n_localities=int(la.size), min=float(la.min()), max=float(la.max()), median_of_medians=float(la.median()),
               overall_median=float(df.Age_of_Property.median()), top5_oldest=la.sort_values(ascending=False).head(5).to_dict(),
               top5_newest=la.sort_values().head(5).to_dict(), share_between=float(((la >= 16) & (la <= 20)).mean() * 100))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
sns.histplot(la, bins=15, ax=ax[0], color=P[4]); ax[0].set(title="Q8 Median property age across 500 localities", xlabel="Median age (years)")
x = pd.concat([la.sort_values().head(10), la.sort_values().tail(10)]); sns.barplot(x=x.values, y=x.index, ax=ax[1], color=P[4]); ax[1].set(title="10 newest & 10 oldest localities", xlabel="Median age"); save("q08_age_locality.png")
# ---- Q9 BHK by city
bc = pd.crosstab(df.City, df.BHK); bp = bc.div(bc.sum(1), axis=0) * 100
chi = stats.chi2_contingency(bc)
R["q9"] = dict(overall=(df.BHK.value_counts(normalize=True).sort_index() * 100).round(1).to_dict(), chi2_p=float(chi[1]), chi2=float(chi[0]),
               min_share=float(bp.values.min()), max_share=float(bp.values.max()), top_city_3bhk=bp[3].sort_values(ascending=False).head(3).round(1).to_dict())
plt.figure(figsize=(10, 9)); sns.heatmap(bp.round(1), annot=True, fmt=".0f", cmap="Blues", cbar_kws={"label": "% of city listings"}); plt.title("Q9 BHK mix by city (% of listings)"); save("q09_bhk_city.png")
# ---- Q10 top-5 expensive localities (city+locality, min 10 listings) + trend by build-year cohort
g = df.groupby(["City", "Locality"]).agg(avg=("Price_in_Lakhs", "mean"), n=("ID", "count")).query("n>=10").sort_values("avg", ascending=False)
top5 = g.head(5); R["q10_top5"] = [dict(city=i[0], locality=i[1], avg_price=float(r.avg), n=int(r.n)) for i, r in top5.iterrows()]
R["q10_overall_locality_avg_range"] = [float(g.avg.min()), float(g.avg.max())]
key = df.set_index(["City", "Locality"]).index.isin(top5.index)
sub = df[key].copy(); sub["Loc"] = sub.City + " / " + sub.Locality
sub["Cohort"] = (sub.Year_Built // 5 * 5).astype(int)
tr = sub.pivot_table(index="Cohort", columns="Loc", values="Price_in_Lakhs", aggfunc="mean")
yr = sub.groupby("Loc").apply(lambda x: stats.spearmanr(x.Year_Built, x.Price_in_Lakhs)[0], include_groups=False)
R["q10_trend_corr"] = yr.round(3).to_dict()
all_tr = df.assign(Cohort=(df.Year_Built // 5 * 5)).groupby("Cohort").Price_in_Lakhs.mean()
R["q10_market_trend"] = all_tr.round(1).to_dict()
plt.figure(figsize=(9, 4.8)); tr.plot(marker="o", ax=plt.gca()); plt.plot(all_tr.index, all_tr.values, "k--", label="All-India avg", lw=2)
plt.title("Q10 Price by build-year cohort, 5 most expensive localities"); plt.ylabel("Avg price (Lakhs)"); plt.xlabel("Year-built cohort"); plt.legend(fontsize=7); save("q10_top_localities_trend.png")
# ---- Q11 correlation
num = ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt_Rs", "BHK", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals", "Floor_No", "Total_Floors", "Amenity_Count", "Transport_Score", "Parking_Flag", "Security_Flag", "Infrastructure_Score"]
cm = df[num].corr().copy(); cm = cm.where(~np.eye(len(num), dtype=bool))
pairs = cm.abs().stack().sort_values(ascending=False).drop_duplicates()
top = [(a, b, float(cm.loc[a, b])) for (a, b) in pairs.index[:8]]
R["q11"] = dict(top_pairs=top, max_abs=float(cm.abs().max().max()),
                derived_excluded_max=float(df[["Price_in_Lakhs", "Size_in_SqFt", "BHK", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals", "Floor_No", "Total_Floors", "Amenity_Count", "Transport_Score", "Parking_Flag", "Security_Flag"]].corr().abs().where(~np.eye(12, dtype=bool)).max().max()))
plt.figure(figsize=(10, 8)); sns.heatmap(df[num].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, annot_kws={"size": 7}); plt.title("Q11 Correlation matrix"); save("q11_correlation.png")
# ---- Q12/Q13 schools & hospitals
for q, c, t in [("q12", "Nearby_Schools", "schools"), ("q13", "Nearby_Hospitals", "hospitals")]:
    m = df.groupby(c).Price_per_SqFt_Rs.mean().round(0)
    R[q] = dict(pearson=float(df[c].corr(df.Price_per_SqFt_Rs)), spearman=spear(c, "Price_per_SqFt_Rs"), by_count=m.to_dict(), anova=anova(c), range_pct=float((m.max() / m.min() - 1) * 100))
    plt.figure(figsize=(7, 4.3)); sns.lineplot(x=m.index, y=m.values, marker="o"); plt.ylim(0, m.max() * 1.25); plt.title(f"{q.upper()} Avg price per sq ft vs nearby {t}"); plt.xlabel(f"Nearby {t}"); plt.ylabel("Rs per sq ft"); save(f"{q}_{t}_ppsf.png")
# ---- Q14 furnished
fs = df.groupby("Furnished_Status").agg(price=("Price_in_Lakhs", "mean"), ppsf=("Price_per_SqFt_Rs", "mean"), n=("ID", "count")).round(1)
R["q14"] = dict(table=fs.reset_index().to_dict("records"), anova_price=anova("Furnished_Status", "Price_in_Lakhs"), anova_ppsf=anova("Furnished_Status"))
fig, ax = plt.subplots(1, 2, figsize=(10, 4)); sns.barplot(x=fs.index, y=fs.price, ax=ax[0], hue=fs.index, legend=False); ax[0].set(title="Q14 Avg price (Lakhs) by furnishing", ylim=(0, fs.price.max() * 1.2))
sns.barplot(x=fs.index, y=fs.ppsf, ax=ax[1], hue=fs.index, legend=False); ax[1].set(title="Avg price per sq ft (Rs)", ylim=(0, fs.ppsf.max() * 1.2)); save("q14_furnished.png")
# ---- Q15 facing
fc = df.groupby("Facing").Price_per_SqFt_Rs.agg(["mean", "median"]).round(0)
R["q15"] = dict(table=fc.reset_index().to_dict("records"), anova=anova("Facing"), range_pct=float((fc["mean"].max() / fc["mean"].min() - 1) * 100))
plt.figure(figsize=(6.5, 4.3)); sns.barplot(x=fc.index, y=fc["mean"], hue=fc.index, legend=False); plt.ylim(0, fc["mean"].max() * 1.2); plt.title("Q15 Avg price per sq ft by facing"); plt.ylabel("Rs per sq ft"); save("q15_facing.png")
# ---- Q16 owner type / Q17 availability
for q, c, ttl in [("q16", "Owner_Type", "Q16 Properties by owner type"), ("q17", "Availability_Status", "Q17 Properties by availability status")]:
    v = df[c].value_counts(); R[q] = dict(counts=v.to_dict(), pct=(v / len(df) * 100).round(2).to_dict(), chi_uniform_p=float(stats.chisquare(v.values)[1]))
    plt.figure(figsize=(6, 4)); ax = sns.barplot(x=v.index, y=v.values, hue=v.index, legend=False); [ax.bar_label(c_, fmt="{:,.0f}") for c_ in ax.containers]; plt.title(ttl); plt.ylim(0, v.max() * 1.15); plt.ylabel("Properties"); save(f"{q}_{c.lower()}.png")
# ---- Q18 parking
pk = df.groupby("Parking_Space").agg(price=("Price_in_Lakhs", "mean"), ppsf=("Price_per_SqFt_Rs", "mean"), n=("ID", "count")).round(1)
tt = stats.ttest_ind(df[df.Parking_Flag == 1].Price_in_Lakhs, df[df.Parking_Flag == 0].Price_in_Lakhs)
R["q18"] = dict(table=pk.reset_index().to_dict("records"), t_p=float(tt.pvalue), diff_pct=float((pk.price["Yes"] / pk.price["No"] - 1) * 100))
plt.figure(figsize=(5.5, 4.2)); sns.barplot(x=pk.index, y=pk.price, hue=pk.index, legend=False); plt.ylim(0, pk.price.max() * 1.2); plt.title("Q18 Avg price (Lakhs) by parking space"); save("q18_parking.png")
# ---- Q19 amenities
am = {a: float(df[df[f"Has_{a}"] == 1].Price_per_SqFt_Rs.mean() / df[df[f"Has_{a}"] == 0].Price_per_SqFt_Rs.mean() * 100 - 100) for a in ["Playground", "Gym", "Garden", "Pool", "Clubhouse"]}
ac = df.groupby("Amenity_Count").Price_per_SqFt_Rs.mean().round(0)
R["q19"] = dict(effect_pct=am, by_count=ac.to_dict(), spearman=spear("Amenity_Count", "Price_per_SqFt_Rs"), anova=anova("Amenity_Count"),
                p_values={a: float(stats.ttest_ind(df[df[f"Has_{a}"] == 1].Price_per_SqFt_Rs, df[df[f"Has_{a}"] == 0].Price_per_SqFt_Rs).pvalue) for a in am})
fig, ax = plt.subplots(1, 2, figsize=(11, 4)); sns.barplot(x=list(am), y=list(am.values()), ax=ax[0], color=P[0]); ax[0].axhline(0, c="k", lw=.8); ax[0].set(title="Q19 % difference in price/sq ft: has vs lacks amenity", ylabel="%")
sns.lineplot(x=ac.index, y=ac.values, marker="o", ax=ax[1]); ax[1].set(title="Avg price/sq ft by # amenities", ylim=(0, ac.max() * 1.25), xlabel="Amenity count", ylabel="Rs/sq ft"); save("q19_amenities.png")
# ---- Q20 transport
tp = df.groupby("Public_Transport_Accessibility").agg(ppsf=("Price_per_SqFt_Rs", "mean"), good=("Good_Investment", "mean"), growth=("Est_Annual_Growth", "mean"), n=("ID", "count"))
tp = tp.loc[["Low", "Medium", "High"]]
R["q20"] = dict(table=tp.reset_index().to_dict("records"), anova_ppsf=anova("Public_Transport_Accessibility"),
                chi_good_p=float(stats.chi2_contingency(pd.crosstab(df.Public_Transport_Accessibility, df.Good_Investment))[1]))
fig, ax = plt.subplots(1, 2, figsize=(10, 4)); sns.barplot(x=tp.index, y=tp.ppsf, ax=ax[0], hue=tp.index, legend=False); ax[0].set(title="Q20 Avg price/sq ft by transport access", ylim=(0, tp.ppsf.max() * 1.2))
sns.barplot(x=tp.index, y=tp.good * 100, ax=ax[1], hue=tp.index, legend=False); ax[1].set(title="% Good Investment (rule-based label)", ylabel="%"); save("q20_transport.png")
# ---- Extra: crime-rate question in brief's EDA list -> no crime column
R["crime_col_present"] = any("crime" in c.lower() for c in df.columns)
# ---- Extra: infra score vs growth / future price; city good-investment share
R["infra_vs_ppsf"] = spear("Infrastructure_Score", "Price_per_SqFt_Rs")
R["infra_vs_price"] = spear("Infrastructure_Score", "Price_in_Lakhs")
R["area_vs_growth"] = spear("Size_in_SqFt", "Est_Annual_Growth")
R["area_vs_good"] = float(df.groupby(pd.qcut(df.Size_in_SqFt, 5)).Good_Investment.mean().max() - df.groupby(pd.qcut(df.Size_in_SqFt, 5)).Good_Investment.mean().min())
gs = df.groupby("State").Good_Investment.mean().sort_values(ascending=False) * 100
R["good_by_state_top3"] = gs.head(3).round(1).to_dict(); R["good_by_state_bottom3"] = gs.tail(3).round(1).to_dict()
gc = df.groupby("City").Good_Investment.mean().sort_values(ascending=False) * 100
R["good_by_city_top3"] = gc.head(3).round(1).to_dict(); R["good_by_city_bottom3"] = gc.tail(3).round(1).to_dict()
R["good_overall"] = float(df.Good_Investment.mean() * 100)
plt.figure(figsize=(7, 4.5)); sns.scatterplot(data=df.sample(6000, random_state=2), x="Infrastructure_Score", y="Price_per_SqFt_Rs", s=7, alpha=.4); plt.yscale("log"); plt.title("Infrastructure score vs price per sq ft (6k sample)"); save("extra_infra_vs_ppsf.png")
plt.figure(figsize=(9, 5)); sns.barplot(x=gc.head(10).values, y=gc.head(10).index, color=P[2]); plt.title("Top 10 cities by % Good Investment (rule-based label)"); plt.xlabel("%"); save("extra_good_by_city.png")
json.dump(R, open("outputs/eda_findings.json", "w"), indent=2, default=float)
print("done", len(list(__import__("glob").glob(F + "*.png"))), "figures")
