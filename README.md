# **🏠 House Price Prediction Dashboard - README**

## **📌 Project Overview**
This interactive Streamlit application provides a comprehensive toolkit for analyzing and predicting house prices using:
- **Exploratory Data Analysis (EDA)**
- **Geospatial Visualization**
- **Machine Learning Models** (Random Forest, Neural Network, SVR)
- **Interactive Prediction Tool**

## **✨ Key Features**

### **📊 Data Exploration**
✅ Interactive scatter plots with trendlines  
✅ Feature distribution visualizations  
✅ Correlation heatmaps  
✅ Outlier detection and removal  

### **🌍 Geographic Insights**
✅ Interactive Folium map with house locations  
✅ Heatmap visualization of price distribution  

### **🤖 Machine Learning Models**
| Model | Features |
|-------|----------|
| **Random Forest** | Feature importance, residual analysis |
| **Neural Network** | Training history, architecture visualization |
| **SVR** | Kernel selection, regularization tuning |

### **🔮 Prediction Tools**
✅ Custom price prediction based on property features  
✅ Similar property recommendations  
✅ Model performance metrics (MAE, RMSE, R²)  

## **⚙️ Installation & Setup**

1. Install requirements:
```bash
pip install streamlit pandas numpy matplotlib seaborn folium scikit-learn tensorflow plotly
```

2. Run the app:
```bash
streamlit run app.py
```

## **🚀 Quick Start Guide**

1. **Upload your dataset** (CSV format with house price data)
2. **Explore the data**:
   - View distributions and correlations
   - Check geographic distribution
3. **Train models**:
   - Select model type in sidebar
   - Adjust hyperparameters
   - View training metrics
4. **Make predictions**:
   - Enter property details
   - Get price estimates

## **📈 Model Performance Comparison**

| Metric | Random Forest | Neural Network | SVR |
|--------|--------------|----------------|-----|
| MAE | $XX,XXX | $XX,XXX | $XX,XXX |
| Training Time | Fast | Medium | Slow |
| Best For | Baseline | Complex patterns | Small datasets |

## **🧠 Educational Value**
- Learn how different ML models perform on real estate data
- Understand feature engineering for housing data
- Visualize model decision processes
- Practice hyperparameter tuning

## **📜 License**
MIT License
---

### **Pro Tips**
1. Try the **"house_age"** feature for interesting price trends
2. Use the **geographic heatmap** to identify premium locations
3. Compare all three models to see which works best for your data

# https://housepricepredictionappml.streamlit.app/ 
