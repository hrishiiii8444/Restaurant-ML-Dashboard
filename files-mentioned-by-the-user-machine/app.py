from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


st.set_page_config(
    page_title="Restaurant ML Dashboard",
    page_icon="ML",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }
    .hero {
        background: linear-gradient(135deg, #111827 0%, #1f2937 55%, #991b1b 100%);
        color: white;
        padding: 28px 30px;
        border-radius: 8px;
        margin-bottom: 18px;
    }
    .hero h1 {
        margin: 0;
        font-size: 34px;
        line-height: 1.15;
    }
    .hero p {
        margin: 10px 0 0 0;
        color: #e5e7eb;
        font-size: 16px;
    }
    .info-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px 18px;
        background: #ffffff;
        box-shadow: 0 1px 8px rgba(15, 23, 42, 0.06);
        min-height: 112px;
    }
    .info-card h3 {
        margin: 0 0 8px 0;
        font-size: 15px;
        color: #374151;
    }
    .info-card p {
        margin: 0;
        font-size: 25px;
        font-weight: 700;
        color: #991b1b;
    }
    .section-note {
        background: #f9fafb;
        border-left: 4px solid #991b1b;
        border-radius: 6px;
        padding: 12px 14px;
        color: #374151;
        margin: 8px 0 18px 0;
    }
    div.stButton > button:first-child {
        border-radius: 6px;
        padding: 0.55rem 1rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_dataset():
    csv_path = Path("Dataset .csv")
    if not csv_path.exists():
        csv_path = Path.home() / "Downloads" / "Dataset .csv"

    if not csv_path.exists():
        st.error("Dataset .csv not found. Keep Dataset .csv in the same folder as app.py.")
        st.stop()

    data = pd.read_csv(csv_path)
    data["Cuisines"] = data["Cuisines"].fillna(data["Cuisines"].mode()[0])
    return data


@st.cache_resource
def train_rating_model(data):
    target = "Aggregate rating"

    columns_to_drop = [
        "Restaurant ID",
        "Restaurant Name",
        "Address",
        "Locality",
        "Locality Verbose",
        "Rating color",
        "Rating text",
        target,
    ]

    X = data.drop(columns=columns_to_drop)
    y = data[target]
    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.25,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }

    importance_df = pd.DataFrame(
        {
            "Feature": X_train.columns,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    return model, X_encoded.columns, metrics, importance_df


@st.cache_resource
def train_recommendation_model(data):
    work = data.copy()

    city_encoder = LabelEncoder()
    work["City_encoded"] = city_encoder.fit_transform(work["City"])

    rating_scaler = MinMaxScaler()
    cost_scaler = MinMaxScaler()
    city_scaler = MinMaxScaler()

    work["Average_Rating_scaled"] = rating_scaler.fit_transform(work[["Aggregate rating"]])
    work["Average_Cost_scaled"] = cost_scaler.fit_transform(work[["Average Cost for two"]])
    work["City_encoded_scaled"] = city_scaler.fit_transform(work[["City_encoded"]])

    feature_columns = [
        "City_encoded_scaled",
        "Average_Rating_scaled",
        "Average_Cost_scaled",
    ]
    X = work[feature_columns].values

    knn = NearestNeighbors(n_neighbors=6, metric="euclidean")
    knn.fit(X)

    return work, X, knn


def build_prediction_input(data, model_columns):
    city = st.selectbox("City", sorted(data["City"].unique()))
    cuisines = st.selectbox("Cuisine", sorted(data["Cuisines"].unique()))
    currency = st.selectbox("Currency", sorted(data["Currency"].unique()))

    col1, col2, col3 = st.columns(3)
    with col1:
        average_cost = st.number_input(
            "Average Cost for Two",
            min_value=0,
            max_value=800000,
            value=500,
            step=50,
        )
    with col2:
        price_range = st.selectbox("Price Range", sorted(data["Price range"].unique()))
    with col3:
        votes = st.number_input("Votes", min_value=0, max_value=20000, value=100, step=10)

    col4, col5, col6 = st.columns(3)
    with col4:
        table_booking = st.selectbox("Has Table Booking", sorted(data["Has Table booking"].unique()))
    with col5:
        online_delivery = st.selectbox("Has Online Delivery", sorted(data["Has Online delivery"].unique()))
    with col6:
        delivering_now = st.selectbox("Is Delivering Now", sorted(data["Is delivering now"].unique()))

    order_menu = st.selectbox("Switch To Order Menu", sorted(data["Switch to order menu"].unique()))

    selected_rows = data[data["City"] == city]
    longitude = float(selected_rows["Longitude"].median()) if not selected_rows.empty else 0.0
    latitude = float(selected_rows["Latitude"].median()) if not selected_rows.empty else 0.0

    if not selected_rows.empty:
        country_code = int(selected_rows["Country Code"].mode()[0])
    else:
        country_code = int(data["Country Code"].mode()[0])

    user_input = pd.DataFrame(
        [
            {
                "Country Code": country_code,
                "City": city,
                "Longitude": longitude,
                "Latitude": latitude,
                "Cuisines": cuisines,
                "Average Cost for two": average_cost,
                "Currency": currency,
                "Has Table booking": table_booking,
                "Has Online delivery": online_delivery,
                "Is delivering now": delivering_now,
                "Switch to order menu": order_menu,
                "Price range": price_range,
                "Votes": votes,
            }
        ]
    )

    encoded_input = pd.get_dummies(user_input, drop_first=True)
    encoded_input = encoded_input.reindex(columns=model_columns, fill_value=0)
    return encoded_input


data = load_dataset()
rating_model, rating_columns, rating_metrics, importance_df = train_rating_model(data)
recommendation_data, recommendation_X, recommendation_model = train_recommendation_model(data)

st.markdown(
    """
    <div class="hero">
        <h1>Restaurant Machine Learning Dashboard</h1>
        <p>Two ML projects in one interface: rating prediction with Random Forest and restaurant recommendation with KNN.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Project Summary")
    st.write("Dataset: Restaurant data")
    st.write("Project 1: Rating prediction")
    st.write("Project 2: Restaurant recommendation")
    st.write("Algorithms: Random Forest, KNN")
    st.divider()
    st.write("Viva point:")
    st.info(
        "The first model is supervised regression. The second model is a similarity-based recommendation system."
    )

overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
with overview_col1:
    st.markdown(
        f'<div class="info-card"><h3>Total Records</h3><p>{data.shape[0]:,}</p></div>',
        unsafe_allow_html=True,
    )
with overview_col2:
    st.markdown(
        f'<div class="info-card"><h3>Total Columns</h3><p>{data.shape[1]}</p></div>',
        unsafe_allow_html=True,
    )
with overview_col3:
    st.markdown(
        f'<div class="info-card"><h3>Cities</h3><p>{data["City"].nunique()}</p></div>',
        unsafe_allow_html=True,
    )
with overview_col4:
    st.markdown(
        f'<div class="info-card"><h3>Average Rating</h3><p>{data["Aggregate rating"].mean():.2f}</p></div>',
        unsafe_allow_html=True,
    )

tab1, tab2, tab3 = st.tabs(
    [
        "Rating Prediction",
        "Restaurant Recommendation",
        "Dataset Overview",
    ]
)

with tab1:
    st.subheader("Predict Restaurant Rating")
    st.markdown(
        '<div class="section-note">Enter restaurant details and the Random Forest model will predict the aggregate rating.</div>',
        unsafe_allow_html=True,
    )

    input_row = build_prediction_input(data, rating_columns)

    if st.button("Predict Rating", type="primary"):
        predicted_rating = rating_model.predict(input_row)[0]
        predicted_rating = float(np.clip(predicted_rating, 0, 5))

        st.success("Prediction completed successfully.")
        st.metric("Predicted Aggregate Rating", f"{predicted_rating:.2f} / 5")

    st.divider()
    st.subheader("Model Performance")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("MAE", f"{rating_metrics['MAE']:.3f}")
    metric_col2.metric("MSE", f"{rating_metrics['MSE']:.3f}")
    metric_col3.metric("R2 Score", f"{rating_metrics['R2']:.3f}")

    st.subheader("Top Feature Importance")
    st.caption("This chart shows which input features influenced the model most.")
    top_features = importance_df.head(10).set_index("Feature")
    st.bar_chart(top_features["Importance"])

with tab2:
    st.subheader("Recommend Similar Restaurants")
    st.markdown(
        '<div class="section-note">Select a restaurant and the KNN model will show restaurants with similar city, rating, and average cost.</div>',
        unsafe_allow_html=True,
    )

    only_rated = st.checkbox("Show only rated restaurants", value=True)
    option_data = recommendation_data.copy()
    if only_rated:
        option_data = option_data[option_data["Aggregate rating"] > 0]

    option_data = option_data.sort_values(["Aggregate rating", "Votes"], ascending=False)
    restaurant_options = option_data["Restaurant Name"].astype(str).unique()
    selected_restaurant = st.selectbox("Select Restaurant", restaurant_options)

    selected_indices = recommendation_data.index[
        recommendation_data["Restaurant Name"].astype(str) == selected_restaurant
    ].tolist()

    if selected_indices:
        selected_index = selected_indices[0]
        selected_details = recommendation_data.loc[
            selected_index,
            ["Restaurant Name", "City", "Cuisines", "Average Cost for two", "Aggregate rating", "Votes"],
        ]

        st.write("Selected Restaurant")
        st.dataframe(selected_details.to_frame("Value"), use_container_width=True)

        if st.button("Recommend Restaurants", type="primary"):
            distances, indices = recommendation_model.kneighbors(
                recommendation_X[selected_index].reshape(1, -1)
            )

            recommended = recommendation_data.iloc[indices[0][1:]].copy()
            recommended["Similarity Distance"] = distances[0][1:]

            if only_rated:
                recommended = recommended[recommended["Aggregate rating"] > 0]

            st.write("Recommended Restaurants")
            st.dataframe(
                recommended[
                    [
                        "Restaurant Name",
                        "City",
                        "Cuisines",
                        "Average Cost for two",
                        "Aggregate rating",
                        "Votes",
                        "Similarity Distance",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

with tab3:
    st.subheader("Dataset Preview")
    st.markdown(
        '<div class="section-note">This section is useful during viva to show that the data was loaded and checked before modeling.</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(data.head(20), use_container_width=True)

    st.subheader("Missing Values")
    missing_df = pd.DataFrame({"Missing Values": data.isnull().sum()})
    st.dataframe(missing_df, use_container_width=True)

    st.subheader("Rating Distribution")
    st.bar_chart(data["Aggregate rating"].value_counts().sort_index())
