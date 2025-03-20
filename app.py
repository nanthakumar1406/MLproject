import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import plotly.express as px
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import time

# Page configuration
# st.set_page_config(layout="wide")

# Load the dataset
df = pd.read_csv("C:/Users/Lenova/OneDrive/ML/deploy/House Price India.csv")

# Drop columns that are not useful for prediction
df.drop(columns=['id', 'Date'], inplace=True, errors='ignore')

# Handle missing values
df.fillna(df.median(), inplace=True)

# Feature Engineering
if 'Built Year' in df.columns:
    df['house_age'] = 2025 - df['Built Year']
if 'Renovation Year' in df.columns:
    df['since_renovation'] = df.apply(lambda row: 2025 - row['Renovation Year'] if row['Renovation Year'] > 0 else 0, axis=1)

# Outlier Detection & Removal using Z-score
num_cols = df.select_dtypes(include=[np.number]).columns
z_scores = np.abs(stats.zscore(df[num_cols]))
df_cleaned = df[(z_scores < 3).all(axis=1)]  # Keep only rows where all features are within 3 std deviations

st.set_page_config(layout="wide")  # Set layout to wide for better side-by-side display
st.title("House Price Prediction")

col1, col2 = st.columns([1, 3])  # Create two columns

with col1:
    st.subheader("User Input for Prediction")
    user_input = {}
    for feature in ['living area', 'lot area', 'Area of the house(excluding basement)',
                    'Area of the basement', 'Number of schools nearby', 'Distance from the airport',
                    'house_age', 'since_renovation']:
        user_input[feature] = st.number_input(f"Enter {feature}", value=0, step=1)

    if st.button("Predict Price"):
        filtered_df = df_cleaned.drop(columns=['Price'], errors='ignore')
        similar_rows = df_cleaned.loc[(filtered_df[list(user_input.keys())] - pd.Series(user_input)).abs().sum(axis=1).idxmin()]
        predicted_price = similar_rows["Price"]
        st.write(f"Predicted Price based on dataset: {predicted_price:.2f}")

with col2:
    st.subheader("Dataset Overview")
    st.write(df_cleaned.head())
    
    # Check for missing values
    missing_values = df_cleaned.isnull().sum()
    st.subheader("Missing Values")
    st.write(missing_values[missing_values > 0])

    st.subheader("Data Visualizations")

    # Histograms for key numerical features
    num_features = ["Price", "living area", "lot area", "number of bedrooms", "number of bathrooms", "number of floors"]
    st.write("### Histograms of Key Features")
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    axes = axes.flatten()
    for i, feature in enumerate(num_features):
        sns.histplot(df_cleaned[feature], bins=30, kde=True, color='blue', ax=axes[i])
        axes[i].set_title(f"Distribution of {feature}")
    plt.tight_layout()
    st.pyplot(fig)
    
    # Correlation Matrix
    st.write("### Correlation Matrix")
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(df_cleaned.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    plt.title("Correlation Heatmap")
    st.pyplot(fig)
    
    # Box Plot to Detect Outliers
    st.write("### Box Plot of House Prices")
    fig, ax = plt.subplots(figsize=(5, 6))
    sns.boxplot(x=df_cleaned["Price"], color="red", ax=ax)
    plt.title("Box Plot of House Prices (Detecting Outliers)")
    st.pyplot(fig)
    
    # Scatter Plot
    st.write("### Scatter Plot: Living Area vs Price")
    fig, ax = plt.subplots(figsize=(5, 6))
    sns.scatterplot(x=df_cleaned['living area'], y=df_cleaned['Price'], ax=ax)
    st.pyplot(fig)
    
    # Bar Chart
    st.write("### Bar Chart: Number of Schools Nearby")
    fig, ax = plt.subplots(figsize=(5, 6))
    df_cleaned['Number of schools nearby'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

   # Selecting features and target variable
    X = df_cleaned.drop(columns=["Price"], errors='ignore')
    y = df_cleaned["Price"]
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Train Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Feature Importance Plot
    feature_importance = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    feature_importance_df = feature_importance.reset_index()
    feature_importance_df.columns = ['Feature', 'Importance']
    fig_fi = px.bar(feature_importance_df, x='Feature', y='Importance', title='Feature Importance (Random Forest)')
    st.plotly_chart(fig_fi)


    # # Live Updating Year-wise Price Trend
    # st.write("### 📊 Live House Price Trends by Year")
    # if 'Built Year' in df.columns:
    #     yearly_price = df.groupby('Built Year')['Price'].mean().reset_index()
    #     fig_trend = px.line(yearly_price, x='Built Year', y='Price', title='Yearly Average House Prices')
    #     st.plotly_chart(fig_trend)


    # Title
    st.write("### 📊 Live House Price Trends by Year")
    
    # Check if 'Built Year' column exists
    if 'Built Year' in df.columns:
        # Group by year and calculate the average price
        yearly_price = df.groupby('Built Year')['Price'].mean().reset_index()
    
        # Sort the data by year
        yearly_price = yearly_price.sort_values(by='Built Year')
    
        # Create a placeholder for the chart
        trend_chart = st.empty()
    
        # Live update animation
        for i in range(1, len(yearly_price) + 1):
            fig_trend = px.line(yearly_price.iloc[:i], x='Built Year', y='Price',
                                title='Yearly Average House Prices',
                                markers=True, line_shape='spline')
    
            trend_chart.plotly_chart(fig_trend)  # Update the chart
            time.sleep(0.5)  # Delay for animation effect


    # Check if Latitude and Longitude columns exist
    if 'Lattitude' in df_cleaned.columns and 'Longitude' in df_cleaned.columns:
        df_cleaned = df_cleaned.dropna(subset=['Lattitude', 'Longitude'])  # Drop missing values
    
        if not df_cleaned.empty:
            # Select a random 500 locations for better performance
            df_sample = df_cleaned.sample(n=min(500, len(df_cleaned)), random_state=42)
    
            # Create a map centered around the average coordinates
            map_center = [df_sample['Lattitude'].mean(), df_sample['Longitude'].mean()]
            m = folium.Map(location=map_center, zoom_start=10)
    
            # Add house locations to the map
            for _, row in df_sample.iterrows():
                folium.Marker([row['Lattitude'], row['Longitude']], popup="House Location").add_to(m)
    
            st.write("### 🗺️ Map Visualizations")
            st_folium(m, width=950, height=700)


# Selecting features and target variable
X = df_cleaned.drop(columns=["Price"], errors='ignore')
y = df_cleaned["Price"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train Neural Network using TensorFlow/Keras
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
nn_model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, verbose=0)



# Evaluate models
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    st.subheader(f"{model_name} Performance")
    st.write(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_nn = nn_model.predict(X_test)

evaluate_model(y_test, y_pred_rf, "Random Forest Regressor")
evaluate_model(y_test, y_pred_nn, "Neural Network Regressor")


        # Random Forest Predictions
y_pred_rf = rf_model.predict(X_test)

# Neural Network Predictions (Flatten predictions for compatibility)
y_pred_nn = nn_model.predict(X_test).flatten()

# Residual Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Random Forest Residuals
sns.residplot(x=y_test, y=y_pred_rf, lowess=True, ax=axes[0], color="blue")
axes[0].set_title("Residual Plot - Random Forest")
axes[0].set_xlabel("Actual Values")
axes[0].set_ylabel("Residuals")

# Neural Network Residuals
sns.residplot(x=y_test, y=y_pred_nn, lowess=True, ax=axes[1], color="green")
axes[1].set_title("Residual Plot - Neural Network")
axes[1].set_xlabel("Actual Values")
axes[1].set_ylabel("Residuals")

plt.tight_layout()
st.pyplot(fig)

# Prediction vs Actual Scatter Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Random Forest Predictions vs Actual
axes[0].scatter(y_test, y_pred_rf, color="blue", alpha=0.6)
axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
axes[0].set_title("Predicted vs Actual - Random Forest")
axes[0].set_xlabel("Actual Values")
axes[0].set_ylabel("Predicted Values")

# Neural Network Predictions vs Actual
axes[1].scatter(y_test, y_pred_nn, color="green", alpha=0.6)
axes[1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
axes[1].set_title("Predicted vs Actual - Neural Network")
axes[1].set_xlabel("Actual Values")
axes[1].set_ylabel("Predicted Values")

plt.tight_layout()
st.pyplot(fig)
