"""
Nassau Candy Distributor — Sales & Profit Optimization Dashboard
------------------------------------------------------------------
Streamlit app converted from the original analysis notebook.

Note on scope: the source dataset's "Ship Date" column is not reliably
linked to "Order Date" (years span 2026-2030 vs 2024-2025), so any
Lead-Time / shipping-speed model built on it would be meaningless.
This app therefore focuses on Sales & Profit insights and a
data-driven allocation recommender instead.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Nassau Candy — Sales & Profit Optimizer",
    page_icon="🍬",
    layout="wide",
)

DATA_PATH = "Nassau Candy Distributor.csv"
CAT_FEATURES = ["Product Name", "Region", "Division", "Ship Mode"]
NUM_FEATURES = ["Units"]


# --------------------------------------------------------------------------
# Data loading & cleaning
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and cleaning data...")
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df = df.dropna().drop_duplicates()
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="mixed", dayfirst=True, errors="coerce")
    df["Margin"] = df["Gross Profit"] / df["Sales"]
    return df


@st.cache_resource(show_spinner="Training models...")
def train_models(df: pd.DataFrame):
    """Train Linear Regression / Random Forest / Gradient Boosting to
    predict order Sales value from product & shipping attributes."""
    X = df[CAT_FEATURES + NUM_FEATURES]
    y = df["Sales"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    results = {}
    pipelines = {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        results[name] = {
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds),
        }
        pipelines[name] = pipe

    best_name = max(results, key=lambda n: results[n]["R2"])
    return results, pipelines, best_name


@st.cache_data(show_spinner=False)
def cluster_region_product(df: pd.DataFrame):
    agg = (
        df.groupby(["Region", "Product Name"])
        .agg(Total_Sales=("Sales", "sum"), Avg_Profit=("Gross Profit", "mean"))
        .reset_index()
    )
    scaler = StandardScaler()
    scaled = scaler.fit_transform(agg[["Total_Sales", "Avg_Profit"]])
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    agg["Cluster"] = km.fit_predict(scaled)

    # Relabel clusters by mean profit so labels are human-readable
    order = agg.groupby("Cluster")["Avg_Profit"].mean().sort_values().index
    label_map = {order[0]: "Underperformer", order[1]: "Steady", order[2]: "Star"}
    agg["Segment"] = agg["Cluster"].map(label_map)
    return agg


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("🍬 Nassau Candy Optimizer")
uploaded = st.sidebar.file_uploader("Upload a different CSV (optional)", type="csv")
df = load_data(uploaded) if uploaded is not None else load_data(DATA_PATH)

st.sidebar.markdown("---")
regions = sorted(df["Region"].unique())
divisions = sorted(df["Division"].unique())
region_filter = st.sidebar.multiselect("Filter by Region", regions, default=regions)
division_filter = st.sidebar.multiselect("Filter by Division", divisions, default=divisions)

view_df = df[df["Region"].isin(region_filter) & df["Division"].isin(division_filter)]

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ The original 'Ship Date' field doesn't line up with 'Order Date' "
    "in this dataset, so this app skips Lead-Time/shipping-speed analysis "
    "and focuses on Sales & Profit instead."
)

# --------------------------------------------------------------------------
# Header / KPIs
# --------------------------------------------------------------------------
st.title("Nassau Candy Distributor — Sales & Profit Optimization")
st.caption("Interactive dashboard for product, region, and division performance.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Sales", f"${view_df['Sales'].sum():,.0f}")
k2.metric("Total Gross Profit", f"${view_df['Gross Profit'].sum():,.0f}")
k3.metric("Avg. Margin", f"{view_df['Margin'].mean() * 100:.1f}%")
k4.metric("Orders", f"{len(view_df):,}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "📈 Sales & Profit Insights", "🤖 Sales Prediction Models", "🎯 Allocation Recommender"]
)

# --------------------------------------------------------------------------
# Tab 1 — Overview
# --------------------------------------------------------------------------
with tab1:
    st.subheader("Data preview")
    st.dataframe(view_df.head(20), width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Summary statistics")
        st.dataframe(view_df[["Sales", "Units", "Gross Profit", "Cost", "Margin"]].describe())
    with c2:
        st.subheader("Records by Division")
        st.dataframe(
            view_df.groupby("Division").size().rename("Orders").reset_index(),
            width='stretch',
        )

# --------------------------------------------------------------------------
# Tab 2 — EDA
# --------------------------------------------------------------------------
with tab2:
    st.subheader("Average Gross Profit by Product")
    profit_by_product = (
        view_df.groupby("Product Name")["Gross Profit"].mean().sort_values(ascending=False).reset_index()
    )
    fig = px.bar(profit_by_product, x="Product Name", y="Gross Profit", color="Gross Profit")
    fig.update_layout(xaxis_tickangle=-40)
    st.plotly_chart(fig, width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Monthly Sales Trend")
        trend = (
            view_df.set_index("Order Date")["Sales"].resample("MS").sum().reset_index()
        )
        st.plotly_chart(px.line(trend, x="Order Date", y="Sales", markers=True), width='stretch')
    with c2:
        st.subheader("Cost vs. Gross Profit")
        st.plotly_chart(
            px.scatter(view_df, x="Cost", y="Gross Profit", color="Division", opacity=0.6),
            width='stretch',
        )

    st.subheader("Profit Margin by Division")
    margin_div = view_df.groupby("Division")["Margin"].mean().reset_index()
    st.plotly_chart(px.bar(margin_div, x="Division", y="Margin", color="Division"), width='stretch')

# --------------------------------------------------------------------------
# Tab 3 — Model comparison
# --------------------------------------------------------------------------
with tab3:
    st.subheader("Predicting order Sales value")
    st.write(
        "Three regression models are trained on **Product, Region, Division, "
        "Ship Mode, and Units** to predict the **Sales value** of an order. "
        "Models are trained once on the full dataset (filters above don't apply here)."
    )
    results, pipelines, best_name = train_models(df)
    metrics_df = pd.DataFrame(results).T.sort_values("R2", ascending=False)
    st.dataframe(metrics_df.style.format("{:.2f}"), width='stretch')
    st.success(f"Best performing model: **{best_name}** (R² = {results[best_name]['R2']:.2f})")

    # Feature importance for tree-based models
    if best_name in ("Random Forest", "Gradient Boosting"):
        pipe = pipelines[best_name]
        feature_names = pipe.named_steps["prep"].get_feature_names_out()
        importances = pipe.named_steps["model"].feature_importances_
        imp_df = (
            pd.DataFrame({"Feature": feature_names, "Importance": importances})
            .sort_values("Importance", ascending=False)
            .head(15)
        )
        st.subheader(f"Top feature importances ({best_name})")
        st.plotly_chart(px.bar(imp_df, x="Importance", y="Feature", orientation="h"), width='stretch')

    st.markdown("---")
    st.subheader("Region × Product performance segments")
    st.write("KMeans clustering of Region/Product combinations by total sales and average profit.")
    seg = cluster_region_product(df)
    st.plotly_chart(
        px.scatter(
            seg, x="Total_Sales", y="Avg_Profit", color="Segment", hover_data=["Region", "Product Name"]
        ),
        width='stretch',
    )
    st.dataframe(seg.sort_values("Avg_Profit", ascending=False), width='stretch')

# --------------------------------------------------------------------------
# Tab 4 — Recommender
# --------------------------------------------------------------------------
with tab4:
    st.subheader("Allocation Recommender")
    st.write(
        "Pick a product to see which **Region + Ship Mode** combination is "
        "predicted to generate the highest order value and profit, based on "
        "historical patterns. (Note: in this dataset each product belongs to "
        "exactly one Division/factory, so Division isn't a free choice — "
        "Region and Ship Mode are the levers a distributor can actually act on.)"
    )

    results, pipelines, best_name = train_models(df)
    model = pipelines[best_name]

    col1, col2 = st.columns(2)
    with col1:
        product = st.selectbox("Product", sorted(df["Product Name"].unique()))
    with col2:
        units = st.number_input(
            "Units per order", min_value=1, value=int(df["Units"].median()), step=1
        )

    product_division = df.loc[df["Product Name"] == product, "Division"].iloc[0]
    avg_margin = df.loc[df["Product Name"] == product, "Margin"].mean()

    combos = pd.DataFrame(
        [
            {"Product Name": product, "Region": r, "Division": product_division, "Ship Mode": m, "Units": units}
            for r in df["Region"].unique()
            for m in df["Ship Mode"].unique()
        ]
    )
    combos["Predicted_Sales"] = model.predict(combos[CAT_FEATURES + NUM_FEATURES])
    combos["Predicted_Profit"] = combos["Predicted_Sales"] * avg_margin
    combos = combos.sort_values("Predicted_Profit", ascending=False).reset_index(drop=True)

    top = combos.iloc[0]
    st.success(
        f"📌 Recommendation: ship **{product}** via **{top['Ship Mode']}** to the "
        f"**{top['Region']}** region — predicted order value ≈ ${top['Predicted_Sales']:.2f}, "
        f"predicted profit ≈ ${top['Predicted_Profit']:.2f}."
    )

    st.dataframe(
        combos[["Region", "Ship Mode", "Predicted_Sales", "Predicted_Profit"]].style.format(
            {"Predicted_Sales": "${:.2f}", "Predicted_Profit": "${:.2f}"}
        ),
        width='stretch',
    )
    st.plotly_chart(
        px.bar(combos, x="Region", y="Predicted_Profit", color="Ship Mode", barmode="group"),
        width='stretch',
    )
