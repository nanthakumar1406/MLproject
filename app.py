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
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go
import time
from streamlit_folium import st_folium
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Title
st.title("🏠 House Price Prediction App")
st.markdown("""
This app analyzes house price data, builds predictive models, and provides interactive visualizations.
""")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    model_type = st.radio(
        "Select Model Type",
        ["Random Forest", "Neural Network","Support Vector Regression"]
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
        return None

df = load_data()

if df is not None:
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
        # Numerical Features Summary
        st.subheader("Numerical Features Summary")
        st.dataframe(df_cleaned.describe().style.format("{:.2f}"))
        
        # Interactive Scatter Plot
        st.subheader("Feature vs Price Analysis")
        col1, col2 = st.columns([1, 3])
        
        with col1:
            selected_feature = st.selectbox(
                "Select feature to plot against price",
                df_cleaned.select_dtypes(include=[np.number]).columns,
                index=0  # Default to first column
            )
            
            # Optional: Add log scale toggle
            log_scale = st.checkbox("Use log scale for Price", True)
        
        with col2:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df_cleaned[selected_feature],
                y=df_cleaned['Price'],
                mode='markers',
                text=df_cleaned.columns,
                name=f"Price vs {selected_feature}",
                marker=dict(size=6, opacity=0.8)
            ))

            fig.update_layout(
                title=f"Price vs {selected_feature}",
                xaxis_title=selected_feature,
                yaxis_title="Price",
            )

            st.plotly_chart(fig)

        # Histograms of Key Features
        st.subheader("Distribution of Key Features")
        num_features = ["Price", "living area", "lot area", "number of bedrooms", 
                       "number of bathrooms", "number of floors"]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, feature in enumerate(num_features):
            if feature in df_cleaned.columns:
                sns.histplot(df_cleaned[feature], bins=30, kde=True, color='blue', ax=axes[i])
                axes[i].set_title(f"Distribution of {feature}")
        plt.tight_layout()
        st.pyplot(fig)
        
        # Box Plot of House Prices
        st.subheader("Price Distribution with Outliers")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x=df_cleaned["Price"], color="red", ax=ax)
        plt.title("Box Plot of House Prices (Detecting Outliers)")
        st.pyplot(fig)
        
        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(df_cleaned.corr(), ax=ax, cmap='coolwarm', annot=True, fmt=".2f")
        st.pyplot(fig)

    # Geographic Heatmap
    if show_map:
        if 'Lattitude' in df_cleaned.columns and 'Longitude' in df_cleaned.columns:
            st.header("Geographic Distribution of Houses")
            df_cleaned = df_cleaned.dropna(subset=['Lattitude', 'Longitude'])
            if not df_cleaned.empty:
                df_sample = df_cleaned.sample(n=min(500, len(df_cleaned)), random_state=42)
                map_center = [df_sample['Lattitude'].mean(), df_sample['Longitude'].mean()]
                m = folium.Map(location=map_center, zoom_start=10)
                
                # Add markers
                for _, row in df_sample.iterrows():
                    folium.Marker(
                        [row['Lattitude'], row['Longitude']],
                        popup=f"Price: ${row['Price']:,.0f}"
                    ).add_to(m)
                
                st_folium(m, width=1200, height=600)
        else:
            st.warning("Latitude and Longitude columns not found for map visualization")


# Live Updating Year-wise Price Trend with Animation
st.write("### 📊 Live House Price Trends by Year (Animated)")
if 'Built Year' in df.columns:
    yearly_price = df.groupby('Built Year')['Price'].mean().reset_index()
    
    # Create the base figure
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
    
    # Create frames for animation
    frames = []
    for year in yearly_price['Built Year'].unique():
        frame_data = yearly_price[yearly_price['Built Year'] <= year]
        frames.append(go.Frame(
            data=[go.Scatter(
                x=frame_data['Built Year'],
                y=frame_data['Price'],
                mode='lines'
            )],
            name=str(year)
        ))
    
    fig_trend.frames = frames
    
    # Add animation controls
    fig_trend.update_layout(
        updatemenus=[dict(
            type="buttons",
            buttons=[ 
                dict(
                    label="▶️ Play",
                    method="animate",
                    args=[None, {
                        "frame": {"duration": 500, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 300}
                    }]
                ),
                dict(
                    label="⏸️ Pause",
                    method="animate",
                    args=[[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }]
                )
            ],
            direction="left",
            pad={"r": 10, "t": 10},
            showactive=False,
            x=0.1,
            xanchor="right",
            y=0,
            yanchor="top"
        )]
    )
    
    # Customize hover template
    fig_trend.update_traces(
        hovertemplate="<b>Year:</b> %{x}<br><b>Avg Price:</b> $%{y:,.2f}<extra></extra>"
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Add additional metrics
    with st.expander("📈 Trend Analysis Metrics"):
        col1, col2, col3 = st.columns(3)
        
        latest_year = yearly_price['Built Year'].max()
        oldest_year = yearly_price['Built Year'].min()
        latest_price = yearly_price[yearly_price['Built Year'] == latest_year]['Price'].values[0]
        oldest_price = yearly_price[yearly_price['Built Year'] == oldest_year]['Price'].values[0]
        growth_rate = ((latest_price - oldest_price) / oldest_price) * 100
        
        with col1:
            st.metric("First Year", oldest_year, f"Price: ${oldest_price:,.2f}")
        with col2:
            st.metric("Latest Year", latest_year, f"Price: ${latest_price:,.2f}")
        with col3:
            st.metric("Total Growth", f"{growth_rate:.2f}%")

# User Prediction Tool
if show_user_prediction:
    st.header("🏡 Custom Price Prediction")
    
    # Get available features
    available_features = [col for col in ['living area', 'lot area', 
                                       'Area of the house(excluding basement)', 
                                       'Area of the basement', 
                                       'Number of schools nearby', 
                                       'Distance from the airport',
                                       'house_age', 'since_renovation'] 
                        if col in df_cleaned.columns]
    
    # Create input form
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
            # Find similar house
            filtered_df = df_cleaned.drop(columns=['Price'], errors='ignore')
            distances = (filtered_df[list(user_input.keys())] - pd.Series(user_input)).abs().sum(axis=1)
            
            if not distances.empty:
                similar_idx = distances.idxmin()
                similar_row = df_cleaned.loc[similar_idx]
                
                st.subheader("Prediction Results")
                if pd.notna(similar_row['Price']):
                    st.success(f"Predicted Price based on similar properties: ${similar_row['Price']:,.2f}")
                    
                    # Show similar properties details
                    with st.expander("Show similar property details"):
                        st.write(pd.DataFrame([similar_row]).transpose())
                else:
                    st.warning("Could not determine price for similar property")
            else:
                st.error("No similar properties found in the dataset")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

# Machine Learning Models
st.header("🤖 Machine Learning Models")

# Prepare data
X = df_cleaned.drop(columns=['Price'])
y = df_cleaned['Price']

# Convert categorical variables if any
X = pd.get_dummies(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=42
)

# Scale data
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
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            end_time = time.time()
            
            st.success(f"Model trained in {end_time - start_time:.2f} seconds")
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"${mae:,.2f}")
            col2.metric("MSE", f"${mse:,.2f}")
            col3.metric("RMSE", f"${rmse:,.2f}")
            col4.metric("R² Score", f"{r2:.4f}")
            
            # Feature importance
            st.subheader("Feature Importance")
            importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            fig = px.bar(importance.reset_index(), x='index', y=0, 
                         labels={'index': 'Feature', '0': 'Importance'},
                         title='Feature Importance (Random Forest)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Residual plot
            st.subheader("Residual Analysis")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.residplot(x=y_test, y=y_pred, lowess=True, color="blue", ax=ax)
            ax.set_title("Residual Plot - Random Forest")
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Residuals")
            st.pyplot(fig)
            
            # Actual vs Predicted
            st.subheader("Actual vs Predicted Prices")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, y_pred, color="blue", alpha=0.6)
            ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
                   color="red", linestyle="--")
            ax.set_title("Predicted vs Actual - Random Forest")
            ax.set_xlabel("Actual Prices")
            ax.set_ylabel("Predicted Prices")
            st.pyplot(fig)

elif model_type == "Support Vector Regression":
    st.subheader("Support Vector Regression (SVR)")

    # Hyperparameter tuning options
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
            0.1, 10.0, 1.0, 
            help="Smaller values = stronger regularization"
        )
    
    if st.button("Train SVR Model"):
        with st.spinner("Training SVR (this may take longer than linear models)..."):
            start_time = time.time()
            
            from sklearn.svm import SVR
            model = SVR(
                kernel=kernel_type,
                C=C_value,
                gamma='scale'  # Automatically handles feature scaling
            )
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            end_time = time.time()
            
            st.success(f"SVR trained in {end_time - start_time:.2f} seconds")
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"${mae:,.2f}")
            col2.metric("MSE", f"${mse:,.2f}")
            col3.metric("RMSE", f"${rmse:,.2f}")
            col4.metric("R² Score", f"{r2:.4f}")
            
            # Feature importance (for linear kernel only)
            if kernel_type == "linear":
                st.subheader("Feature Importance (Linear Kernel)")
                importance = pd.DataFrame({
                    "Feature": X.columns,
                    "Coefficient": model.coef_[0]
                }).sort_values("Coefficient", key=abs, ascending=False)
                
                fig = px.bar(
                    importance,
                    x="Feature",
                    y="Coefficient",
                    title="Feature Coefficients (SVR Linear Kernel)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Feature importance only available for linear kernel")
            
            # Residual plot
            st.subheader("Residual Analysis")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.residplot(x=y_test, y=y_pred, lowess=True, color="orange", ax=ax)
            ax.set_title("Residual Plot - SVR")
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Residuals")
            st.pyplot(fig)
            
            # Actual vs Predicted
            st.subheader("Actual vs Predicted Prices")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, y_pred, color="orange", alpha=0.6)
            ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
                   color="red", linestyle="--")
            ax.set_title("Predicted vs Actual - SVR")
            ax.set_xlabel("Actual Prices")
            ax.set_ylabel("Predicted Prices")
            st.pyplot(fig)

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
            
            # Predictions
            y_pred = model.predict(X_test_scaled).flatten()
            
            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            end_time = time.time()
            
            st.success(f"Model trained in {end_time - start_time:.2f} seconds")
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"${mae:,.2f}")
            col2.metric("MSE", f"${mse:,.2f}")
            col3.metric("RMSE", f"${rmse:,.2f}")
            col4.metric("R² Score", f"{r2:.4f}")
            
            # Plot training history
            st.subheader("Training History")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            ax1.plot(history.history['mae'], label='Train MAE')
            ax1.plot(history.history['val_mae'], label='Validation MAE')
            ax1.set_title('MAE Over Epochs')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('MAE')
            ax1.legend()
            
            ax2.plot(history.history['loss'], label='Train Loss')
            ax2.plot(history.history['val_loss'], label='Validation Loss')
            ax2.set_title('Loss Over Epochs')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            
            st.pyplot(fig)
            
            # Residual plot
            st.subheader("Residual Analysis")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.residplot(x=y_test, y=y_pred, lowess=True, color="green", ax=ax)
            ax.set_title("Residual Plot - Neural Network")
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Residuals")
            st.pyplot(fig)
            
            # Actual vs Predicted
            st.subheader("Actual vs Predicted Prices")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, y_pred, color="green", alpha=0.6)
            ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
                   color="red", linestyle="--")
            ax.set_title("Predicted vs Actual - Neural Network")
            ax.set_xlabel("Actual Prices")
            ax.set_ylabel("Predicted Prices")
            st.pyplot(fig)
