import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load dataset
df = pd.read_csv("Housing.csv")

df.describe() 

# Apply log transformation to the target variable
df['price'] = np.log1p(df['price'])

# Encode categorical features
categorical_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea", "furnishingstatus"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Splitting dataset
X = df_encoded.drop(columns=["price"])
y = df_encoded["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Train Neural Network Model
nn_model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])

nn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

nn_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_data=(X_test_scaled, y_test), callbacks=callbacks, verbose=0)

# Model Evaluation
rf_r2 = r2_score(y_test, rf_model.predict(X_test))
nn_r2 = r2_score(y_test, nn_model.predict(X_test_scaled))

# Streamlit UI Setup
st.set_page_config(page_title="🏡 House Price Prediction Dashboard", layout="wide")

# Sidebar for user input
st.sidebar.header("📌 Enter House Details")

area = st.sidebar.slider("🏠 Area (sq. ft.)", 500, 20000, 1000)
bedrooms = st.sidebar.slider("🛏 Bedrooms", 1, 10, 2)
bathrooms = st.sidebar.slider("🛁 Bathrooms", 1, 5, 1)
stories = st.sidebar.slider("📏 Stories", 1, 4, 1)
parking = st.sidebar.slider("🚗 Parking Spaces", 0, 5, 1)

# Categorical inputs
mainroad = st.sidebar.radio("🛣 Main Road?", ["yes", "no"])
guestroom = st.sidebar.radio("🛋 Guest Room?", ["yes", "no"])
basement = st.sidebar.radio("🏠 Basement?", ["yes", "no"])
hotwaterheating = st.sidebar.radio("🔥 Hot Water Heating?", ["yes", "no"])
airconditioning = st.sidebar.radio("❄ Air Conditioning?", ["yes", "no"])
prefarea = st.sidebar.radio("📍 Preferred Location?", ["yes", "no"])
furnishingstatus = st.sidebar.radio("🛏 Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

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
if st.sidebar.button("🚀 Predict Price"):
    prediction_rf = np.expm1(rf_model.predict(input_df))[0]
    prediction_nn = np.expm1(nn_model.predict(input_df_scaled)[0][0])
    st.sidebar.success(f"🏠 Estimated Price (Random Forest): ₹{int(prediction_rf):,}")
    # st.sidebar.success(f"🤖 Estimated Price (Neural Network): ₹{int(prediction_nn):,}")

# Show Model Performance
# st.subheader("📊 Model Performance")
# st.metric(label="Random Forest R² Score", value=f"{rf_r2 * 100:.2f}%")
# st.metric(label="Neural Network R² Score", value=f"{nn_r2 * 100:.2f}%")

# Data Visualizations
st.subheader("📊 Data Analysis & Visualizations")

fig1 = px.histogram(df, x="price", nbins=30, color_discrete_sequence=["#4CAF50"], title="House Price Distribution")
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.scatter(df, x="area", y="price", color="price", size="price", title="House Area vs Price")
st.plotly_chart(fig2, use_container_width=True)

fig3 = px.box(df, x="furnishingstatus", y="price", color="furnishingstatus", title="Price by Furnishing Status")
st.plotly_chart(fig3, use_container_width=True)

fig4 = px.bar(df, x="bedrooms", y="price", color="price", title="Price Distribution by Bedrooms")
st.plotly_chart(fig4, use_container_width=True)

fig5 = px.line(df, x="area", y="price", title="Price Trend by Area")
st.plotly_chart(fig5, use_container_width=True)

fig_pie = px.pie(df, names='furnishingstatus', title='Furnishing Status Proportion')
st.plotly_chart(fig_pie, use_container_width=True)

fig_area = px.histogram(df, x='area', title='Distribution of House Areas', nbins=30)
st.plotly_chart(fig_area, use_container_width=True)

st.subheader("📊 Parallel Coordinates Chart for Feature Relationships")
fig_par = px.parallel_coordinates(df_encoded, dimensions=["area", "bedrooms", "bathrooms", "stories", "price"],
                                  color="price", color_continuous_scale=px.colors.sequential.Viridis)
st.plotly_chart(fig_par, use_container_width=True)

st.subheader("📊 Feature Correlation")
fig_corr, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df_encoded.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig_corr)

st.subheader("📊 Model Accuracy Comparison")
fig_acc = px.bar(x=["Random Forest", "Neural Network"], y=[rf_r2 * 100, nn_r2 * 100],
                 labels={'x': 'Model', 'y': 'R² Score (%)'},
                 title="Comparison of Model Accuracy", color=["Random Forest", "Neural Network"])
st.plotly_chart(fig_acc, use_container_width=True)
# Show dataset (optional)
if st.checkbox("📜 Show Dataset"):
    st.write(df.head())

# Footer
st.markdown("---")
st.markdown("<h4 style='text-align: center;'>📌 Developed with ❤️ using Streamlit</h4>", unsafe_allow_html=True)
