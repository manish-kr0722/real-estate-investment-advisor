"""Builds the project documentation (Word) from outputs/*.json + figures."""
import json
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
R = json.load(open("outputs/eda_findings.json")); M = json.load(open("outputs/model_results.json")); L = json.load(open("outputs/preprocess_log.json"))
F = "outputs/figures/"
doc = Document()
st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10.5)
for s in doc.sections: s.left_margin = s.right_margin = Inches(.9); s.top_margin = s.bottom_margin = Inches(.8)
def H(t, l=1): doc.add_heading(t, l)
def P(t, b=False, i=False):
    p = doc.add_paragraph(); r = p.add_run(t); r.bold = b; r.italic = i; return p
def B(t, lead=None):
    p = doc.add_paragraph(style="List Bullet")
    if lead: p.add_run(lead).bold = True
    p.add_run(t)
def IMG(f, w=6.2):
    doc.add_picture(F + f, width=Inches(w)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
def T(rows, head, widths=None):
    t = doc.add_table(rows=1, cols=len(head)); t.style = "Light Grid Accent 1"
    for i, h in enumerate(head): t.rows[0].cells[i].text = h; t.rows[0].cells[i].paragraphs[0].runs[0].bold = True
    for r in rows:
        c = t.add_row().cells
        for i, v in enumerate(r): c[i].text = str(v)
    doc.add_paragraph()
def Q(n, q, ans, fig=None, w=6.2):
    H(f"Q{n}. {q}", 3); P(ans)
    if fig: IMG(fig, w)
pv = lambda p: "p < 0.001" if p < .001 else f"p = {p:.2f}"
n = lambda x, d=0: f"{x:,.{d}f}"

# ---------------- title
t = doc.add_heading("Real Estate Investment Advisor", 0)
P("Predicting Property Profitability & Future Value - Project Documentation", b=True)
P(f"Dataset: india_housing_prices.csv ({L['rows_raw']:,} listings, 23 columns, 20 states, 42 cities)  |  Prepared by: Manish Kumar", i=True)

H("1. Executive summary")
B(f" {L['rows_raw']:,} records, no missing values, no duplicates. Price per sq ft was recomputed in rupees because the source column (lakhs, 2 decimals) loses precision; 46.5% of rows have Floor_No greater than Total_Floors (flagged, not deleted).", "Data quality:")
B(" Prices, sizes, BHK, age, schools, hospitals, owner type, availability and every location/amenity attribute are spread almost uniformly, and none of them explains price. State, city, property type, furnishing, facing, parking, amenities, schools, hospitals and transport all move price per sq ft by less than about 3%, mostly with no statistical significance. This pattern is typical of synthetic data, so the analysis honestly reports 'no relationship' for most questions instead of inventing insights.", "Main EDA finding:")
B(" The only strong relationship is mechanical: price per sq ft falls as size rises (r = %.2f) because price is independent of size." % R["q4"]["pearson_ppsf"], "Strong relationship:")
B(f" Good Investment (price/sq ft at or below the city-and-type median AND estimated growth >= 7%/yr) is {R['good_overall']:.1f}% of listings. The 5-year price applies the estimated growth rate for 5 years. The dataset has no historical prices, so both targets are rule-based assumptions.", "Targets:")
cm = M["runs"]; bc = [r for r in cm if r["model"] == M["best_classifier"] and r["task"] == "classification"][0]["metrics"]; br = [r for r in cm if r["model"] == M["best_regressor"] and r["task"] == "regression"][0]["metrics"]
B(f" {M['best_classifier']} classifier: accuracy {bc['accuracy']:.1%}, F1 {bc['f1']:.3f}, ROC-AUC {bc['roc_auc']:.4f}. {M['best_regressor']} regressor: RMSE {br['rmse']:.2f} lakhs, R2 {br['r2']:.4f}. These scores are high because the labels are built from the same inputs by rules, so the models recover the rules; they do not prove real-world predictive power.", "Models:")
B(" Streamlit app (prediction form, filters, heatmap and trend charts, confidence score, feature importance) and an MLflow-enabled training script.", "Deliverables:")

H("2. Business use cases")
for a, b in [("Investors: ", "assess long-term return potential of a listing and compare asking price with the city median."), ("Buyers: ", "shortlist properties in developing (under-construction) areas with strong expected growth."), ("Real-estate companies: ", "automate first-pass investment scoring for new listings."), ("Platforms: ", "show transparent, data-backed ratings that build customer trust.")]: B(b, a)

H("3. Methodology")
H("3.1 Data preprocessing", 2)
T([["Missing values", f"{L['missing_cells_raw']} found; script still imputes (median / mode) so it is reusable on other extracts"],
   ["Duplicates", f"{L['duplicate_rows_raw']} duplicate rows, {L['duplicate_ids_raw']} duplicate IDs"],
   ["Price per sq ft", "Recomputed as Price_in_Lakhs x 1e5 / Size_in_SqFt (Rs per sq ft); matches source within rounding"],
   ["Age of property", "Verified = 2025 - Year_Built for every row"],
   ["Floor consistency", f"{L['floor_gt_total_floors_pct']}% of rows have Floor_No > Total_Floors; flagged in Floor_Inconsistent, Floor_Ratio capped at 1"],
   ["Outliers (IQR 1.5x)", f"Price/sq ft: {L['outliers_Price_per_SqFt_Rs']:,} flagged (kept, flagged in Outlier_ columns); size: {L['outliers_Size_in_SqFt']}; price: {L['outliers_Price_in_Lakhs']}"],
   ["Scaling", "StandardScaler on numeric features inside the model pipeline (fit on training data only)"],
   ["Encoding", "One-hot for State, City, Property_Type, Furnished_Status, Facing, Owner_Type; ordinal for transport (Low=1..High=3); binary flags for parking, security, availability; multi-label split of Amenities into 5 flags + count. Locality (500 values reused across cities) and ID are not used in models."]],
  ["Step", "What was done"])
H("3.2 Feature engineering", 2)
P("Price_per_SqFt_Rs, Price_per_BHK, Size_per_BHK, Floor_Ratio, Amenity_Count and 5 amenity flags, School_Density_Score and Hospital_Density_Score (0-10), Transport_Score, Infrastructure_Score (0-100: schools 20, hospitals 20, transport 25, amenities 20, parking 7.5, security 7.5), City_Median_PPSF and Value_Ratio (price/sq ft divided by the median of the same city and property type).")
H("3.3 Target definitions (domain rules)", 2)
P("The dataset contains no historical or future prices, so both targets are derived from assumptions that are stated openly and can be changed at the top of src/preprocess.py:")
B(" 5% base + up to 3% from Infrastructure_Score + 1% if under construction (developing-area premium) + 0.5% if built in the last 10 years - 0.5% if older than 25 years. Range in data: 4.7% to 9.5% per year.", "Estimated annual growth =")
B(" Price_in_Lakhs x (1 + growth)^5. Mean = %s lakhs vs %s lakhs today." % (n(M["mean_future_price"], 1), n(R["q1"]["mean"], 1)), "Future price (regression target) =")
B(f" 1 if Value_Ratio <= 1 (not above the city/type median price per sq ft) AND growth >= 7%. Class balance: {R['good_overall']:.1f}% positive / {100-R['good_overall']:.1f}% negative.", "Good_Investment (classification target) =")
P("Side effect to be aware of: because price per sq ft falls with size, larger properties look 'cheaper per sq ft' and are labelled Good Investment more often (a %.0f-point gap between the smallest and largest size quintile)." % (R["area_vs_good"] * 100), i=True)

H("4. Exploratory Data Analysis - answers to all 20 questions")
P("All figures are in outputs/figures/. Statistical tests: ANOVA / t-test for group differences, Spearman or Pearson for relationships. With 250,000 rows even tiny differences can reach p < 0.05, so practical size (the % difference) is reported alongside.", i=True)
H("4.1 Price & size analysis (Q1-5)", 2)
q = R["q1"]; Q(1, "What is the distribution of property prices?", f"Prices range from {q['min']:.0f} to {q['max']:.0f} lakhs with mean {n(q['mean'],1)}, median {n(q['median'],1)} and standard deviation {n(q['std'],1)}. The distribution is essentially flat (uniform) and symmetric (skewness {q['skew']:.2f}); IQR is {n(q['q25'],0)}-{n(q['q75'],0)} lakhs. There is no concentration around a typical price band.", "q01_price_distribution.png")
q = R["q2"]; Q(2, "What is the distribution of property sizes?", f"Sizes run from {q['min']:.0f} to {q['max']:.0f} sq ft (mean {n(q['mean'])}, median {n(q['median'])}, sd {n(q['std'])}). Again uniform and symmetric (skewness {q['skew']:.3f}).", "q02_size_distribution.png")
q = R["q3"]; rows = [[r["Property_Type"], n(r["mean"]), n(r["median"]), n(r["count"])] for r in q["table"]]
Q(3, "How does price per sq ft vary by property type?", f"It does not. Mean price per sq ft is about Rs 13,000 for Apartments, Independent Houses and Villas (differences under 1%; ANOVA {pv(q['anova']['p'])}). The mean is well above the median (about Rs 9,200) because the distribution is right-skewed: small properties have very high price per sq ft.", "q03_ppsf_property_type.png")
T(rows, ["Property type", "Mean Rs/sq ft", "Median Rs/sq ft", "Listings"])
q = R["q4"]; Q(4, "Is there a relationship between property size and price?", f"No. Size and total price are uncorrelated (Pearson r = {q['pearson']:.3f}, {pv(q['spearman']['p'])}): a 5,000 sq ft property is on average priced like a 500 sq ft one. As a consequence price per sq ft falls strongly with size (r = {q['pearson_ppsf']:.2f}); this is arithmetic, not a market signal. In real markets one would expect a clear positive relationship, which is another sign the data is synthetic.", "q04_size_vs_price.png")
q = R["q5"]; Q(5, "Are there any outliers in price per sq ft or property size?", f"Size and total price have no outliers (IQR rule; both uniformly bounded at 500-5,000 sq ft and 10-500 lakhs). Price per sq ft has {q['ppsf']['n_outliers']:,} outliers ({q['ppsf']['pct']:.1f}%) above Rs {n(q['ppsf']['upper'])}/sq ft, up to Rs {n(q['ppsf_max'])} (median Rs {n(q['ppsf_median'])}; skewness {q['ppsf_skew']:.1f}). They come from small, high-priced listings (e.g. 500 sq ft at 500 lakhs). They are valid records, so they were flagged (Outlier_Price_per_SqFt_Rs) rather than removed.", "q05_outliers.png")
H("4.2 Location-based analysis (Q6-10)", 2)
q = R["q6"]; Q(6, "What is the average price per sq ft by state?", f"Highest: " + ", ".join(f"{r['State']} (Rs {n(r['mean'])})" for r in q["top5"][:3]) + f". Lowest: " + ", ".join(f"{r['State']} (Rs {n(r['mean'])})" for r in q["bottom5"][-3:]) + f". The gap between highest and lowest state is only {q['spread_pct']:.1f}% and is not statistically significant (ANOVA {pv(q['anova']['p'])}), so state does not differentiate price per sq ft in this data.", "q06_ppsf_state.png", 5.2)
q = R["q7"]; Q(7, "What is the average property price by city?", f"City averages lie between {min(q['bottom5'].values()):.1f} lakhs ({min(q['bottom5'], key=q['bottom5'].get)}) and {max(q['top5'].values()):.1f} lakhs ({max(q['top5'], key=q['top5'].get)}) around the overall mean of {q['overall']:.1f}. The spread is {q['spread_pct']:.1f}% and not significant (ANOVA {pv(q['anova']['p'])}). Metros such as Mumbai and New Delhi are not more expensive than smaller cities here, which contradicts real-world patterns.", "q07_price_city.png", 5.0)
q = R["q8"]; Q(8, "What is the median age of properties by locality?", f"Across the {q['n_localities']} locality names the median age is {q['median_of_medians']:.0f} years (same as the overall median), and every locality's median falls between {q['min']:.0f} and {q['max']:.0f} years; {q['share_between']:.1f}% lie between 16 and 20. Oldest: Locality_72 and Locality_403 (21 yrs); newest: Locality_443 (16 yrs). Locality names repeat across cities, so each name pools about 500 listings from many cities.", "q08_age_locality.png")
q = R["q9"]; Q(9, "How is BHK distributed across cities?", f"BHK 1-5 each make up about 20% of listings overall, and the mix is nearly identical in every city: each BHK share per city ranges only from {q['min_share']:.1f}% to {q['max_share']:.1f}%. A chi-square test finds no city-BHK association ({pv(q['chi2_p'])}).", "q09_bhk_city.png", 5.6)
Q(10, "What are the price trends for the top 5 most expensive localities?", "No date column exists, so a true price time-series is impossible. As a proxy, Year_Built cohorts (5-year bands) were used. The 5 most expensive localities (city + locality, minimum 10 listings) are: " + "; ".join(f"{r['city']} / {r['locality']} ({r['avg_price']:.0f} lakhs, n={r['n']})" for r in R["q10_top5"]) + ". Their lines zig-zag with no consistent trend (rank correlation of price with build year ranges from %.2f to %.2f), and each point rests on only 1-5 listings. The all-India line is flat (%.0f-%.0f lakhs across cohorts). These localities rank top largely by chance from small samples; this is not evidence of a persistent premium." % (min(R["q10_trend_corr"].values()), max(R["q10_trend_corr"].values()), min(R["q10_market_trend"].values()), max(R["q10_market_trend"].values())), "q10_top_localities_trend.png", 5.6)
H("4.3 Feature relationships & correlation (Q11-15)", 2)
q = R["q11"]; Q(11, "How are numeric features correlated with each other?", "Among the original columns, correlations are essentially zero (largest absolute value %.3f). The notable correlations involve derived features only: " % q["derived_excluded_max"] + "; ".join(f"{a} vs {b} ({v:+.2f})" for a, b, v in q["top_pairs"][:3]) + ". Infrastructure_Score correlates with its own components (transport, schools, hospitals, amenities), and price per sq ft with size and price by construction. Raw features are independent of one another.", "q11_correlation.png", 6.0)
q = R["q12"]; Q(12, "How do nearby schools relate to price per sq ft?", f"No relationship. Spearman rho = {q['spearman']['r']:.3f} ({pv(q['spearman']['p'])}); average price per sq ft stays between Rs 12,888 and 13,292 for 1-10 schools (range {q['range_pct']:.1f}%, ANOVA {pv(q['anova']['p'])}).", "q12_schools_ppsf.png", 4.6)
q = R["q13"]; Q(13, "How do nearby hospitals relate to price per sq ft?", f"No practical relationship. Spearman rho = {q['spearman']['r']:.3f} ({pv(q['spearman']['p'])}). ANOVA gives {pv(q['anova']['p'])}, but the group means have no pattern (range {q['range_pct']:.1f}%, non-monotonic), which is expected by chance when testing many variables on 250,000 rows.", "q13_hospitals_ppsf.png", 4.6)
q = R["q14"]; Q(14, "How does price vary by furnished status?", "Not at all: " + ", ".join(f"{r['Furnished_Status']} {r['price']:.1f} lakhs (Rs {n(r['ppsf'])}/sq ft)" for r in q["table"]) + f". Differences are below 0.3% (ANOVA on price {pv(q['anova_price']['p'])}; on price/sq ft {pv(q['anova_ppsf']['p'])}). Furnishing is not priced in.", "q14_furnished.png", 5.2)
q = R["q15"]; Q(15, "How does price per sq ft vary by property facing direction?", "West-facing is highest (Rs %s) and East lowest (Rs %s), a gap of %.1f%% that is not significant (ANOVA %s). Facing has no effect here." % (n(max(r['mean'] for r in q['table'])), n(min(r['mean'] for r in q['table'])), q["range_pct"], pv(q["anova"]["p"])), "q15_facing.png", 4.4)
H("4.4 Investment, amenities & ownership (Q16-20)", 2)
q = R["q16"]; Q(16, "How many properties belong to each owner type?", f"Broker {q['counts']['Broker']:,} ({q['pct']['Broker']}%), Owner {q['counts']['Owner']:,} ({q['pct']['Owner']}%), Builder {q['counts']['Builder']:,} ({q['pct']['Builder']}%): an even three-way split (not different from uniform, {pv(q['chi_uniform_p'])}).", "q16_owner_type.png", 4.2)
q = R["q17"]; Q(17, "How many properties are available under each availability status?", f"Under Construction {q['counts']['Under_Construction']:,} ({q['pct']['Under_Construction']}%) and Ready to Move {q['counts']['Ready_to_Move']:,} ({q['pct']['Ready_to_Move']}%); an even split ({pv(q['chi_uniform_p'])}). The data has no 'Sold' status even though the data dictionary lists one.", "q17_availability_status.png", 4.2)
q = R["q18"]; t = {r["Parking_Space"]: r for r in q["table"]}; Q(18, "Does parking space affect property price?", f"No. Average price is {t['Yes']['price']:.1f} lakhs with parking vs {t['No']['price']:.1f} without (+{q['diff_pct']:.2f}%, t-test {pv(q['t_p'])}). Price per sq ft is also the same (Rs {n(t['Yes']['ppsf'])} vs Rs {n(t['No']['ppsf'])}). Parking is a binary Yes/No in this data, not a count.", "q18_parking.png", 4.0)
q = R["q19"]; Q(19, "How do amenities affect price per sq ft?", "Having a given amenity changes price per sq ft by only " + ", ".join(f"{k} {v:+.2f}%" for k, v in q["effect_pct"].items()) + f"; none is significant at 5% (smallest p = {min(q['p_values'].values()):.2f}, Gym). By number of amenities (1-5) price per sq ft is flat (Spearman {q['spearman']['r']:.3f}, {pv(q['spearman']['p'])}; ANOVA {pv(q['anova']['p'])}). Amenities do not add to price per sq ft in this dataset.", "q19_amenities.png")
q = R["q20"]; tp = {r["Public_Transport_Accessibility"]: r for r in q["table"]}
Q(20, "How does public transport accessibility relate to price per sq ft or investment potential?", f"Price per sq ft is unrelated to transport access (Low Rs {n(tp['Low']['ppsf'])}, Medium Rs {n(tp['Medium']['ppsf'])}, High Rs {n(tp['High']['ppsf'])}; ANOVA {pv(q['anova_ppsf']['p'])}). Investment potential, however, does rise with transport: the share of Good Investments is {tp['Low']['good']:.1%} (Low), {tp['Medium']['good']:.1%} (Medium) and {tp['High']['good']:.1%} (High), and estimated annual growth {tp['Low']['growth']:.2%} / {tp['Medium']['growth']:.2%} / {tp['High']['growth']:.2%}. This follows directly from the label rules (transport contributes 25 of the 100 infrastructure points), so it is a design assumption and not an observed market effect.", "q20_transport.png", 5.2)
H("4.5 Additional questions from the brief", 2)
B(f" The dataset has no crime-rate column, so this cannot be analysed. Adding a crime index by locality is the recommended data enhancement.", "Impact of crime rate on Good Investment:")
B(f" Size vs estimated growth: rho = {R['area_vs_growth']['r']:.3f} (none). Larger properties are labelled Good Investment more often only because their price per sq ft is lower (see 3.3).", "Area vs investment return:")
B(f" Infrastructure_Score vs price per sq ft rho = {R['infra_vs_ppsf']['r']:.3f}; vs price rho = {R['infra_vs_price']['r']:.3f}. No relationship with observed resale value; infrastructure only drives the assumed growth rate.", "Infrastructure score vs resale value:")
B(f" Share of Good Investment by state ranges {min(R['good_by_state_bottom3'].values()):.1f}%-{max(R['good_by_state_top3'].values()):.1f}% and by city {min(R['good_by_city_bottom3'].values()):.1f}%-{max(R['good_by_city_top3'].values()):.1f}%: no location is meaningfully better.", "Where are the good investments:")
IMG("extra_infra_vs_ppsf.png", 4.6)

H("5. Modelling")
P(f"Split: 80% train ({M['n_train']:,}) / 20% test ({M['n_test']:,}), stratified on the classification target, random_state = 42. Features: {len(__import__('features').FEATURES)} (see src/features.py). The rule inputs Est_Annual_Growth, Infrastructure_Score, Value_Ratio and City_Median_PPSF are not given to the models, so they must learn the rules from raw attributes.")
H("5.1 Classification: Is it a Good Investment?", 2)
T([[r["model"], f"{r['metrics']['accuracy']:.4f}", f"{r['metrics']['precision']:.4f}", f"{r['metrics']['recall']:.4f}", f"{r['metrics']['f1']:.4f}", f"{r['metrics']['roc_auc']:.4f}"] for r in M["runs"] if r["task"] == "classification"], ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
P(f"Baseline: always predicting 'not good' gives {M['majority_baseline_accuracy']:.1%} accuracy. Best model: {M['best_classifier']} (confusion matrix [[TN, FP], [FN, TP]] = {M['confusion_matrix']}).")
H("5.2 Regression: price after 5 years (lakhs)", 2)
T([[r["model"], f"{r['metrics']['rmse']:.3f}", f"{r['metrics']['mae']:.3f}", f"{r['metrics']['r2']:.4f}", f"{r['metrics']['mape_pct']:.2f}%"] for r in M["runs"] if r["task"] == "regression"], ["Model", "RMSE", "MAE", "R2", "MAPE"])
P(f"Best model: {M['best_regressor']}. Even Ridge reaches R2 0.9986 because the future price is the current price times a growth factor of about 1.26-1.57, so the current price explains almost everything.")
IMG("model_diagnostics.png", 6.2)
H("5.3 Feature importance (permutation importance on the test set)", 2)
P("Classifier: price per sq ft (value test), under-construction status, year built, transport, schools, hospitals, amenity count, parking and security (the growth-rule inputs). Regressor: current price dominates, followed by under-construction, transport, year built, schools, hospitals and amenities. The importance ranking matches the rules used to build the targets, which confirms the models learned them.")
IMG("model_importance_clf.png", 4.4); IMG("model_importance_reg.png", 4.4)
H("5.4 Experiment tracking", 2)
P(f"src/train.py logs every run (parameters, metrics, fitted pipeline) to MLflow with experiments RealEstate_classification and RealEstate_regression; view them with 'mlflow ui'. XGBoost models are trained when the library is installed. In the environment used to generate these results neither XGBoost nor MLflow could be installed, so the scikit-learn HistGradientBoosting model was the boosted model and the run details are stored in outputs/model_results.json. Re-running train.py on a machine with both libraries adds the XGBoost runs and MLflow logs without code changes.")

H("6. Streamlit application")
for a, b in [("Predict tab: ", "form for state, city, type, BHK, size, price, year built, furnishing, floors, schools, hospitals, transport, parking, security, facing, owner, availability and amenities. Returns 'Good Investment?' with confidence, the estimated price after 5 years with implied yearly growth, and the price per sq ft against the city median."), ("Browse & filter tab: ", "filter listings by state, property type, BHK, price range, area and good-investment flag."), ("Market insights tab: ", "city x property-type heatmap of price per sq ft, build-year price trend, good-investment share by state, and the label definitions."), ("Model info tab: ", "metrics for each model and feature-importance charts.")]: B(b, a)
P("Run: streamlit run app.py. The prediction logic was tested outside Streamlit (feature rebuild matches the cleaned dataset; 99.4% accuracy and 0.85-lakh mean error on a fresh sample). The Streamlit user interface itself could not be launched in the build environment, so run it once locally and check the layout before submitting.", i=True)

H("7. Findings, recommendations & limitations")
H("Findings", 2)
for x in ["No location, property-type or amenity attribute in this dataset explains price or price per sq ft; every group average is within about 3% of the overall average.", "Size does not change total price, so smaller properties cost far more per sq ft; a buyer comparing price per sq ft will always find larger properties 'cheaper'.", "Under the stated rules, the best-scoring listings are those that are under construction, close to high public-transport access, with more schools, hospitals and amenities, priced below the city median per sq ft."]: B(x)
H("Recommendations", 2)
for x in ["Replace the assumed growth rule with observed data: multi-year prices or sale/resale records per locality, plus crime, infrastructure projects and rental yield.", "Validate on real listings (for example from a portal) before using the scores for real decisions.", "Add a time column so true price trends (Q10) and genuine appreciation labels can be built."]: B(x)
H("Limitations", 2)
for x in ["Targets are rule-based assumptions, so metrics measure how well rules are recovered, not future market performance.", "The data shows signs of being synthetic (uniform distributions, no location effects, size unrelated to price, 46.5% of floors above building height).", "XGBoost, MLflow and the Streamlit UI were not executed in the build environment (see 5.4 and 6).", "Not financial advice."]: B(x)

H("8. How to reproduce")
P("pip install -r requirements.txt; python src/preprocess.py; python src/eda.py; python src/train.py; mlflow ui; streamlit run app.py. See README.md.")
doc.save("outputs/Project_Documentation.docx"); print("saved")
