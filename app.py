import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import plotly.express as px
import time
from streamlit_folium import st_folium
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction App")
st.markdown("""
This app analyzes house price data, builds predictive models, and provides interactive visualizations.
""")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    model_type = st.radio(
        "Select Model Type",
        ["Random Forest", "Neural Network", "Support Vector Regression"]
    )
    test_size = st.slider("Test Set Size (%)", 10, 40, 20)
    show_eda = st.checkbox("Show Exploratory Data Analysis", True)
    show_map = st.checkbox("Show Geographic Heatmap", False)
    show_user_prediction = st.checkbox("Show Price Prediction Tool", True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("House Price India.csv")
        df.drop(columns=['id', 'Date'], inplace=True, errors='ignore')
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please upload your data file.")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df.drop(columns=['id', 'Date'], inplace=True, errors='ignore')
            return df
        return None

df = load_data()

if df is not None:
    # Rename columns if needed for consistency
    if "Lattitude" in df.columns:
        df.rename(columns={"Lattitude": "Latitude"}, inplace=True)
    # Feature Engineering
    if 'Built Year' in df.columns:
        df['house_age'] = 2025 - df['Built Year']
    if 'Renovation Year' in df.columns:
        df['since_renovation'] = df.apply(
            lambda row: 2025 - row['Renovation Year'] if row['Renovation Year'] > 0 else 0, 
            axis=1
        )
    # Outlier Detection & Removal using Z-score
    num_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(df[num_cols]))
    df_cleaned = df[(z_scores < 3).all(axis=1)]

    # Data Overview
    st.header("Data Overview")
    with st.expander("Show raw data"):
        st.dataframe(df_cleaned)

    # Missing values
    missing_values = df_cleaned.isnull().sum()
    if missing_values.sum() > 0:
        st.warning("Missing Values Detected")
        st.dataframe(missing_values[missing_values > 0])
    else:
        st.success("No missing values found!")

    # EDA Section
    if show_eda:
        st.header("Exploratory Data Analysis")
        st.subheader("Numerical Features Summary")
        st.dataframe(df_cleaned.describe().style.format("{:.2f}"))

        st.subheader("Feature vs Price Analysis")
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_feature = st.selectbox(
                "Select feature to plot against price",
                [c for c in df_cleaned.select_dtypes(include=[np.number]).columns if c != "Price"],
                index=0
            )
            log_scale = st.checkbox("Use log scale for Price", True)
        with col2:
            fig = px.scatter(
                df_cleaned,
                x=selected_feature,
                y='Price',
                trendline="ols",
                hover_data=df_cleaned.columns,
                title=f"Price vs {selected_feature}"
            )
            if log_scale:
                fig.update_yaxes(type="log")
            fig.update_layout(
                hovermode='closest',
                showlegend=False,
                xaxis_title=selected_feature,
                yaxis_title="Price (USD)"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Distribution of Key Features")
        num_features = [c for c in ["Price", "living area", "lot area", "number of bedrooms", "number of bathrooms", "number of floors"] if c in df_cleaned.columns]
        if num_features:
            fig, axes = plt.subplots(1, len(num_features), figsize=(5 * len(num_features), 5))
            if len(num_features) == 1:
                axes = [axes]
            for i, feature in enumerate(num_features):
                sns.histplot(df_cleaned[feature], bins=30, kde=True, color='blue', ax=axes[i])
                axes[i].set_title(f"Distribution of {feature}")
            st.pyplot(fig)

        st.subheader("Price Distribution with Outliers")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x=df_cleaned["Price"], color="red", ax=ax)
        plt.title("Box Plot of House Prices (Detecting Outliers)")
        st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(df_cleaned.corr(), ax=ax, cmap='coolwarm', annot=True, fmt=".2f")
        st.pyplot(fig)

    # Geographic Heatmap
    if show_map:
        if 'Latitude' in df_cleaned.columns and 'Longitude' in df_cleaned.columns:
            st.header("Geographic Distribution of Houses")
            df_cleaned_map = df_cleaned.dropna(subset=['Latitude', 'Longitude'])
            if not df_cleaned_map.empty:
                df_sample = df_cleaned_map.sample(n=min(500, len(df_cleaned_map)), random_state=42)
                map_center = [df_sample['Latitude'].mean(), df_sample['Longitude'].mean()]
                m = folium.Map(location=map_center, zoom_start=10)
                for _, row in df_sample.iterrows():
                    folium.Marker(
                        [row['Latitude'], row['Longitude']],
                        popup=f"Price: ${row['Price']:,.0f}"
                    ).add_to(m)
                st_folium(m, width=1200, height=600)
        else:
            st.warning("Latitude and Longitude columns not found for map visualization")

    # Live Updating Year-wise Price Trend with Animation
    st.write("### 📊 Live House Price Trends by Year (Animated)")
    if 'Built Year' in df.columns and 'Price' in df.columns:
        yearly_price = df.groupby('Built Year')['Price'].mean().reset_index()
        fig_trend = go.Figure(
            data=[go.Scatter(
                x=yearly_price['Built Year'],
                y=yearly_price['Price'],
                mode='lines',
                line=dict(width=3, color='#1f77b4')
            )],
            layout=go.Layout(
                title='<b>Yearly Average House Prices</b>',
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(count=5, label="5y", step="year", stepmode="backward"),
                            dict(count=10, label="10y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    title='Year'
                ),
                yaxis=dict(
                    title='Average Price (USD)',
                    fixedrange=False
                ),
                hovermode="x unified",
                template='plotly_white'
            )
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # User Prediction Tool
    if show_user_prediction:
        st.header("🏡 Custom Price Prediction")
        available_features = [col for col in [
            'living area', 'lot area', 'Area of the house(excluding basement)', 'Area of the basement',
            'Number of schools nearby', 'Distance from the airport', 'house_age', 'since_renovation'
        ] if col in df_cleaned.columns]
        with st.form("user_input_form"):
            user_input = {}
            cols = st.columns(2)
            for i, feature in enumerate(available_features):
                with cols[i % 2]:
                    user_input[feature] = st.number_input(
                        f"Enter {feature}",
                        min_value=float(df_cleaned[feature].min()),
                        max_value=float(df_cleaned[feature].max()),
                        value=float(df_cleaned[feature].median())
                    )
            submitted = st.form_submit_button("Predict Price")
        if submitted:
            try:
                filtered_df = df_cleaned.drop(columns=['Price'], errors='ignore')
                distances = (filtered_df[list(user_input.keys())] - pd.Series(user_input)).abs().sum(axis=1)
                if not distances.empty:
                    similar_idx = distances.idxmin()
                    similar_row = df_cleaned.loc[similar_idx]
                    st.subheader("Prediction Results")
                    if pd.notna(similar_row['Price']):
                        st.success(f"Predicted Price based on similar properties: ${similar_row['Price']:,.2f}")
                        with st.expander("Show similar property details"):
                            st.write(pd.DataFrame([similar_row]).transpose())
                    else:
                        st.warning("Could not determine price for similar property")
                else:
                    st.error("No similar properties found in the dataset")
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")

    # Modeling Section
    st.header("🤖 Machine Learning Models")
    X = df_cleaned.drop(columns=['Price'])
    y = df_cleaned['Price']
    X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "Random Forest":
        st.subheader("Random Forest Regression")
        n_estimators = st.slider("Number of trees", 10, 200, 100)
        max_depth = st.slider("Max depth", 1, 20, 10)
        if st.button("Train Random Forest Model"):
            with st.spinner("Training model..."):
                start_time = time.time()
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                end_time = time.time()
                st.success(f"Model trained in {end_time - start_time:.2f} seconds")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MAE", f"${mae:,.2f}")
                col2.metric("MSE", f"${mse:,.2f}")
                col3.metric("RMSE", f"${rmse:,.2f}")
                col4.metric("R² Score", f"{r2:.4f}")

    elif model_type == "Support Vector Regression":
        st.subheader("Support Vector Regression (SVR)")
        col1, col2 = st.columns(2)
        with col1:
            kernel_type = st.selectbox(
                "Kernel",
                ["rbf", "linear", "poly", "sigmoid"],
                index=0
            )
        with col2:
            C_value = st.slider(
                "Regularization (C)",
                0.1, 10.0, 1.0
            )
        if st.button("Train SVR Model"):
            with st.spinner("Training SVR model..."):
                start_time = time.time()
                model = SVR(
                    kernel=kernel_type,
                    C=C_value,
                    gamma='scale'
                )
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                end_time = time.time()
                st.success(f"SVR trained in {end_time - start_time:.2f} seconds")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MAE", f"${mae:,.2f}")
                col2.metric("MSE", f"${mse:,.2f}")
                col3.metric("RMSE", f"${rmse:,.2f}")
                col4.metric("R² Score", f"{r2:.4f}")

    else:  # Neural Network
        st.subheader("Neural Network Regression")
        epochs = st.slider("Epochs", 10, 200, 50)
        batch_size = st.slider("Batch size", 16, 128, 32)
        if st.button("Train Neural Network"):
            with st.spinner("Training model..."):
                start_time = time.time()
                model = Sequential([
                    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                    Dense(32, activation='relu'),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                early_stop = EarlyStopping(monitor='val_loss', patience=5)
                history = model.fit(
                    X_train_scaled, y_train,
                    validation_split=0.2,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stop],
                    verbose=0
                )
                y_pred = model.predict(X_test_scaled).flatten()
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                end_time = time.time()
                st.success(f"Model trained in {end_time - start_time:.2f} seconds")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MAE", f"${mae:,.2f}")
                col2.metric("MSE", f"${mse:,.2f}")
                col3.metric("RMSE", f"${rmse:,.2f}")
                col4.metric("R² Score", f"{r2:.4f}")

st.markdown("---")
st.markdown("""
**Note:** This is a demo application. For accurate predictions, ensure your data is clean and properly preprocessed.
""")
