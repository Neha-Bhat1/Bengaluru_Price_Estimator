import streamlit as st, pandas as pd, joblib
import plotly.express as px

st.set_page_config(page_title="Bengaluru Home Price Estimator", layout="wide")

@st.cache_resource
def load():
    return joblib.load("model.joblib"), joblib.load("locations.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("clean.csv")

model, locations = load()
data = load_data()

st.title("Bengaluru Home Price Estimator")
st.caption("Estimates are indicative, not valuations.")

with st.sidebar:
    st.header("Your property")
    loc = st.selectbox("Location", locations)
    sqft = st.number_input("Area (sqft)", 300, 6000, 1200, step=50)
    bhk = st.slider("Bedrooms (BHK)", 1, 6, 2)
    bath = st.slider("Bathrooms", 1, 6, 2)

row = pd.DataFrame([{"location": loc, "total_sqft": sqft, "bath": bath, "bhk": bhk}])
pred = model.predict(row)[0]

c1, c2, c3 = st.columns(3)
c1.metric("Estimated price", f"₹{pred:.0f} lakh")
c2.metric("Price per sqft", f"₹{pred*100000/sqft:,.0f}")
c3.metric(f"Typical in {loc}", f"₹{data[data.location == loc]['pps'].median():,.0f}/sqft")

st.subheader("How does size change the price here?")
sizes = list(range(max(300, sqft - 600), sqft + 601, 100))
curve = pd.DataFrame({"location": loc, "total_sqft": sizes, "bath": bath, "bhk": bhk})
curve["price"] = model.predict(curve)
st.plotly_chart(px.line(curve, x="total_sqft", y="price",
                labels={"total_sqft": "Area (sqft)", "price": "Price (lakh ₹)"}),
                use_container_width=True)

top = data.groupby("location")["pps"].median().nlargest(10).reset_index()
st.subheader("Top 10 priciest areas")
st.plotly_chart(px.bar(top, x="pps", y="location", orientation="h"),
                use_container_width=True)
asking = st.sidebar.number_input("Asking price (lakh ₹), optional", 0, 5000, 0, step=5)

if asking > 0:
    diff = (asking - pred) / pred * 100
    if diff > 15:
        st.warning(f"Asking price is about {diff:.0f}% above our estimate. Worth negotiating.")
    elif diff < -15:
        st.success(f"Asking price is about {abs(diff):.0f}% below our estimate. Check why.")
    else:
        st.info("Asking price is within a fair range of our estimate.")
import numpy as np
from scipy.sparse import issparse

X_t = model.named_steps["prep"].transform(row)
if issparse(X_t):
    X_t = X_t.toarray()
tree_preds = [t.predict(X_t)[0] for t in model.named_steps["model"].estimators_]
low, high = np.percentile(tree_preds, [10, 90])
st.caption(f"Likely range: ₹{low:.0f} to ₹{high:.0f} lakh")
st.subheader("Where can I afford?")
budget = st.number_input("Your budget (lakh ₹)", 20, 1000, 80, step=5)

all_locs = pd.DataFrame({"location": locations, "total_sqft": sqft, "bath": bath, "bhk": bhk})
all_locs["estimate"] = model.predict(all_locs)
affordable = all_locs[all_locs["estimate"] <= budget].sort_values("estimate", ascending=False)

st.write(f"{len(affordable)} areas fit a {bhk} BHK of {sqft} sqft within your budget.")
st.dataframe(affordable[["location", "estimate"]].rename(
    columns={"location": "Area", "estimate": "Estimated price (lakh ₹)"}),
    hide_index=True)
loc2 = st.sidebar.selectbox("Compare with", ["None"] + locations)
if loc2 != "None":
    row2 = row.copy(); row2["location"] = loc2
    pred2 = model.predict(row2)[0]
    st.metric(f"{loc2} for the same property", f"₹{pred2:.0f} lakh",
              delta=f"{pred2 - pred:+.0f} lakh vs {loc}")
