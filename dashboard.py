import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("Housing.csv")

# Apply log transformation to the target variable
df['price'] = np.log1p(df['price'])

# Encode categorical features
categorical_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea", "furnishingstatus"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Splitting dataset
X = df_encoded.drop(columns=["price"])
y = df_encoded["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features for SVR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train Support Vector Regression Model
svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train)

# Predictions
rf_pred_log = rf_model.predict(X_test)
svr_pred_log = svr_model.predict(X_test_scaled)

# Reverse log transformation
y_pred_rf = np.expm1(rf_pred_log)
y_pred_svr = np.expm1(svr_pred_log)
y_test_original = np.expm1(y_test)

# Evaluate models
rf_r2 = r2_score(y_test_original, y_pred_rf) * 100
svr_r2 = r2_score(y_test_original, y_pred_svr) * 100
rf_test_score = rf_model.score(X_test, y_test) * 100
svr_test_score = svr_model.score(X_test_scaled, y_test) * 100

# Streamlit UI
st.set_page_config(page_title="🏡 House Price Prediction Dashboard", layout="wide")
st.title("🏡 House Price Prediction Dashboard")
st.write("Analyze house prices and make predictions based on features.")

# Sidebar for user input
st.sidebar.header("📌 Enter House Details")
area = st.sidebar.number_input("Area (sq. ft.)", min_value=500, max_value=20000, value=1000)
bedrooms = st.sidebar.number_input("Bedrooms", min_value=1, max_value=10, value=2)
bathrooms = st.sidebar.number_input("Bathrooms", min_value=1, max_value=5, value=1)
stories = st.sidebar.number_input("Stories", min_value=1, max_value=4, value=1)
parking = st.sidebar.number_input("Parking Spaces", min_value=0, max_value=5, value=1)

# Categorical inputs
mainroad = st.sidebar.selectbox("Main Road?", ["yes", "no"])
guestroom = st.sidebar.selectbox("Guest Room?", ["yes", "no"])
basement = st.sidebar.selectbox("Basement?", ["yes", "no"])
hotwaterheating = st.sidebar.selectbox("Hot Water Heating?", ["yes", "no"])
airconditioning = st.sidebar.selectbox("Air Conditioning?", ["yes", "no"])
prefarea = st.sidebar.selectbox("Preferred Location?", ["yes", "no"])
furnishingstatus = st.sidebar.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

# Encode user inputs
feature_values = {
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "parking": parking,
    "mainroad_yes": 1 if mainroad == "yes" else 0,
    "guestroom_yes": 1 if guestroom == "yes" else 0,
    "basement_yes": 1 if basement == "yes" else 0,
    "hotwaterheating_yes": 1 if hotwaterheating == "yes" else 0,
    "airconditioning_yes": 1 if airconditioning == "yes" else 0,
    "prefarea_yes": 1 if prefarea == "yes" else 0,
    "furnishingstatus_semi-furnished": 1 if furnishingstatus == "semi-furnished" else 0,
    "furnishingstatus_unfurnished": 1 if furnishingstatus == "unfurnished" else 0
}

# Create input dataframe
input_df = pd.DataFrame([feature_values])
input_df_scaled = scaler.transform(input_df)

# Prediction
if st.sidebar.button("Predict Price 💰"):
    prediction_rf = np.expm1(rf_model.predict(input_df))[0]
    prediction_svr = np.expm1(svr_model.predict(input_df_scaled))[0]
    st.sidebar.success(f"🏠  Estimated Price: ₹{int(prediction_rf):,}")
    st.sidebar.success(f"🏠 SVR Estimated Price: ₹{int(prediction_svr):,}")

# Model Performance Dashboard
st.header("📊 Model Performance Dashboard Random Forest")
st.write(f"**🔹  Train Score:** {rf_test_score:.2f}%")
st.write(f"**🔹  R² Score:** {rf_r2:.2f}%")
st.write(f"**🔹 SVR Test Score:** {svr_test_score:.2f}%")
st.write(f"**🔹 SVR R² Score:** {svr_r2:.2f}%")

# Data Visualizations
st.header("📊 Data Analysis & Visualizations")
st.subheader("House Price Distribution")
fig1, ax1 = plt.subplots(figsize=(8, 4))
sns.histplot(df["price"], bins=30, kde=True, color="blue", ax=ax1)
st.pyplot(fig1)

st.subheader("Feature Correlation Heatmap")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax2)
st.pyplot(fig2)

st.subheader("House Area vs Price")
fig3 = px.scatter(df, x="area", y="price", color="price", size="price", title="House Area vs Price")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Price Variation by Furnishing Status")
fig4 = px.box(df, x="furnishingstatus", y="price", color="furnishingstatus", title="Price by Furnishing Status")
st.plotly_chart(fig4, use_container_width=True)

st.subheader("Price Distribution by Number of Bedrooms")
fig5 = px.bar(df, x="bedrooms", y="price", color="price", title="Price Distribution by Bedrooms")
st.plotly_chart(fig5, use_container_width=True)

st.subheader("Single Line Chart for Price Trend")
fig6 = px.line(df, x="area", y="price", title="Price Trend by Area")
st.plotly_chart(fig6, use_container_width=True)

# Show dataset (optional)
if st.checkbox("Show Dataset 📜"):
    st.write(df.head())

# Footer
st.markdown("---")
st.markdown("📌 **Developed with ❤️ using Streamlit**")
