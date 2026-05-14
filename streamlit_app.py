# ============================================================
#        Car Price Prediction Dashboard - Streamlit App
#        CodeAlpha Data Science Internship - Task 3
#        Student: Aiman | ID: CA/DF1/54987
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Car Price Prediction Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS - ULTRA DARK THEME
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(to bottom right, #0b0618, #140b2d, #1c103f);
    color: white;
}

/* Main Title */
.main-title {
    font-size: 52px;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(90deg, #ff4da6, #9b59f5, #5bbfde);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

/* Subtitle */
.sub-title {
    text-align: center;
    color: #d0cde1;
    font-size: 18px;
    margin-bottom: 40px;
}

/* Metric Cards */
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: 0.3s;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0px 0px 20px rgba(155, 89, 245, 0.4);
}

/* Section Heading */
.section-heading {
    font-size: 30px;
    font-weight: 700;
    color: #ffffff;
    margin-top: 20px;
    margin-bottom: 15px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #12091f;
}

/* Tables */
[data-testid="stDataFrame"] {
    border-radius: 15px;
    overflow: hidden;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #ff4da6, #9b59f5);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 25px;
    font-size: 16px;
    font-weight: 600;
}

.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0px 0px 15px rgba(255, 77, 166, 0.5);
}

/* Prediction Box */
.pred-box {
    background: linear-gradient(135deg, #ff4da6, #9b59f5);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    color: white;
    font-size: 26px;
    font-weight: 700;
    margin-top: 20px;
}

/* Insights Box */
.insight-box {
    background: rgba(255,255,255,0.05);
    border-left: 5px solid #ff4da6;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# TITLE
# ============================================================

st.markdown('<div class="main-title">🚗 Car Price Prediction Dashboard</div>', unsafe_allow_html=True)

st.markdown("""
<div class="sub-title">
Advanced Machine Learning Dashboard using Linear Regression & Random Forest Regressor
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("car data.csv")
    df.columns = df.columns.str.strip()
    return df

car = load_data()

# ============================================================
# DATA CLEANING
# ============================================================

duplicates = car.duplicated().sum()
if duplicates > 0:
    car.drop_duplicates(inplace=True)

# Feature Engineering
car['Brand'] = car['Car_Name'].str.split().str[0].str.lower()

car['Brand'] = car['Brand'].replace({
    'vw': 'volkswagen'
})

car['Car_Age'] = 2024 - car['Year']

# Drop columns
car.drop(['Car_Name', 'Year'], axis=1, inplace=True)

# ============================================================
# OUTLIER HANDLING
# ============================================================

outlier_cols = ['Selling_Price', 'Present_Price', 'Driven_kms']

for col in outlier_cols:
    Q1 = car[col].quantile(0.25)
    Q3 = car[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    car[col] = car[col].clip(lower=lower, upper=upper)

# ============================================================
# LABEL ENCODING
# ============================================================

le = LabelEncoder()

car['Fuel_Type'] = le.fit_transform(car['Fuel_Type'])
car['Selling_type'] = le.fit_transform(car['Selling_type'])
car['Transmission'] = le.fit_transform(car['Transmission'])
car['Brand'] = le.fit_transform(car['Brand'])

# ============================================================
# TRAIN MODEL
# ============================================================

X = car.drop('Selling_Price', axis=1)
y = car['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

r2_lr = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

# Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

best_model = "Random Forest" if r2_rf > r2_lr else "Linear Regression"

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🚘 Dashboard Navigation")

section = st.sidebar.radio(
    "Go To",
    [
        "Overview",
        "Dataset",
        "EDA & Graphs",
        "Model Performance",
        "Feature Importance",
        "Prediction System",
        "Key Insights"
    ]
)

# ============================================================
# OVERVIEW
# ============================================================

if section == "Overview":

    st.markdown('<div class="section-heading">📊 Project Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
        <h2>{car.shape[0]}</h2>
        <p>Total Cars</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
        <h2>{car.shape[1]}</h2>
        <p>Total Features</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
        <h2>{max(r2_lr, r2_rf)*100:.2f}%</h2>
        <p>Best Accuracy</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
        <h2>{best_model}</h2>
        <p>Best Model</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    st.image(
        "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7",
        use_container_width=True
    )

# ============================================================
# DATASET
# ============================================================

elif section == "Dataset":

    st.markdown('<div class="section-heading">📁 Dataset Information</div>', unsafe_allow_html=True)

    st.subheader("Dataset Preview")
    st.dataframe(car.head(15), use_container_width=True)

    st.subheader("Dataset Shape")
    st.write(f"Rows: {car.shape[0]}")
    st.write(f"Columns: {car.shape[1]}")

    st.subheader("Missing Values")
    st.dataframe(car.isnull().sum().to_frame("Missing Values"))

    st.subheader("Statistical Summary")
    st.dataframe(car.describe())

# ============================================================
# EDA & GRAPHS
# ============================================================

elif section == "EDA & Graphs":

    st.markdown('<div class="section-heading">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    # ========================================================
    # Graph 1
    # ========================================================

    fig1, ax1 = plt.subplots(figsize=(10,5))
    fig1.patch.set_facecolor('#140b2d')
    ax1.set_facecolor('#1b1036')

    ax1.hist(
        car['Selling_Price'],
        bins=30,
        color='#9b59f5',
        edgecolor='white'
    )

    ax1.set_title(
        'Selling Price Distribution',
        color='white',
        fontsize=16,
        fontweight='bold'
    )

    ax1.tick_params(colors='white')

    st.pyplot(fig1)

    # ========================================================
    # Graph 2
    # ========================================================

    fig2, ax2 = plt.subplots(figsize=(10,7))
    fig2.patch.set_facecolor('#140b2d')

    sns.heatmap(
        car.corr(),
        annot=True,
        cmap='RdPu',
        ax=ax2
    )

    ax2.set_title(
        'Correlation Heatmap',
        fontsize=16,
        fontweight='bold'
    )

    st.pyplot(fig2)

    # ========================================================
    # Graph 3
    # ========================================================

    fig3, ax3 = plt.subplots(figsize=(8,5))
    fig3.patch.set_facecolor('#140b2d')
    ax3.set_facecolor('#1b1036')

    fuel_avg = car.groupby('Fuel_Type')['Selling_Price'].mean()

    ax3.bar(
        fuel_avg.index.astype(str),
        fuel_avg.values,
        color=['#ff4da6', '#9b59f5', '#5bbfde']
    )

    ax3.set_title(
        'Average Selling Price by Fuel Type',
        color='white',
        fontsize=16,
        fontweight='bold'
    )

    ax3.tick_params(colors='white')

    st.pyplot(fig3)

    # ========================================================
    # Graph 4
    # ========================================================

    fig4, ax4 = plt.subplots(figsize=(9,5))
    fig4.patch.set_facecolor('#140b2d')
    ax4.set_facecolor('#1b1036')

    ax4.scatter(
        car['Present_Price'],
        car['Selling_Price'],
        color='#ff4da6',
        alpha=0.7
    )

    ax4.set_title(
        'Present Price vs Selling Price',
        color='white',
        fontsize=16,
        fontweight='bold'
    )

    ax4.tick_params(colors='white')

    st.pyplot(fig4)

    # ========================================================
    # Graph 5
    # ========================================================

    fig5, ax5 = plt.subplots(figsize=(9,5))
    fig5.patch.set_facecolor('#140b2d')
    ax5.set_facecolor('#1b1036')

    ax5.scatter(
        car['Driven_kms'],
        car['Selling_Price'],
        color='#5bbfde',
        alpha=0.7
    )

    ax5.set_title(
        'Driven KMs vs Selling Price',
        color='white',
        fontsize=16,
        fontweight='bold'
    )

    ax5.tick_params(colors='white')

    st.pyplot(fig5)

    # ========================================================
    # Graph 6
    # ========================================================

    fig6, ax6 = plt.subplots(figsize=(9,5))
    fig6.patch.set_facecolor('#140b2d')
    ax6.set_facecolor('#1b1036')

    ax6.scatter(
        car['Car_Age'],
        car['Selling_Price'],
        color='#f5a623',
        alpha=0.7
    )

    ax6.set_title(
        'Car Age vs Selling Price',
        color='white',
        fontsize=16,
        fontweight='bold'
    )

    ax6.tick_params(colors='white')

    st.pyplot(fig6)

# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif section == "Model Performance":

    st.markdown('<div class="section-heading">🤖 Model Performance</div>', unsafe_allow_html=True)

    comparison_df = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "R2 Score": [r2_lr, r2_rf],
        "MAE": [mae_lr, mae_rf],
        "RMSE": [rmse_lr, rmse_rf]
    })

    st.dataframe(comparison_df, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Linear Regression R2", f"{r2_lr*100:.2f}%")

    with col2:
        st.metric("Random Forest R2", f"{r2_rf*100:.2f}%")

    st.success(f"🏆 Best Model: {best_model}")

    # Actual vs Predicted Graph

    fig7, ax7 = plt.subplots(figsize=(8,6))
    fig7.patch.set_facecolor('#140b2d')
    ax7.set_facecolor('#1b1036')

    ax7.scatter(
        y_test,
        y_pred_rf,
        color='#9b59f5',
        alpha=0.7
    )

    min_val = min(y_test.min(), y_pred_rf.min())
    max_val = max(y_test.max(), y_pred_rf.max())

    ax7.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle='--',
        linewidth=2,
        color='#ff4da6'
    )

    ax7.set_title(
        'Actual vs Predicted Prices',
        color='white',
        fontsize=16,
        fontweight='bold'
    )

    ax7.tick_params(colors='white')

    st.pyplot(fig7)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elif section == "Feature Importance":

    st.markdown('<div class="section-heading">🔥 Feature Importance</div>', unsafe_allow_html=True)

    importances = rf_model.feature_importances_

    feat_imp = pd.Series(
        importances,
        index=X.columns
    ).sort_values()

    fig8, ax8 = plt.subplots(figsize=(10,6))

    fig8.patch.set_facecolor('#140b2d')
    ax8.set_facecolor('#1b1036')

    ax8.barh(
        feat_imp.index,
        feat_imp.values,
        color='#9b59f5'
    )

    ax8.set_title(
        'Feature Importance - Random Forest',
        color='white',
        fontsize=16,
        fontweight='bold'
    )

    ax8.tick_params(colors='white')

    st.pyplot(fig8)

# ============================================================
# PREDICTION SYSTEM
# ============================================================

elif section == "Prediction System":

    st.markdown('<div class="section-heading">🚘 Predict Car Price</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        present_price = st.number_input(
            "Present Price (Lakhs)",
            min_value=0.0,
            value=7.5
        )

        driven_kms = st.number_input(
            "Driven Kilometers",
            min_value=0,
            value=40000
        )

        fuel_type = st.selectbox(
            "Fuel Type",
            ["CNG", "Diesel", "Petrol"]
        )

    with col2:

        selling_type = st.selectbox(
            "Selling Type",
            ["Dealer", "Individual"]
        )

        transmission = st.selectbox(
            "Transmission",
            ["Automatic", "Manual"]
        )

        owner = st.selectbox(
            "Owner",
            [0, 1, 2, 3]
        )

        brand = st.slider(
            "Brand Encoded Value",
            0,
            30,
            10
        )

        car_age = st.slider(
            "Car Age",
            0,
            20,
            5
        )

    # Encoding

    fuel_map = {
        "CNG": 0,
        "Diesel": 1,
        "Petrol": 2
    }

    sell_map = {
        "Dealer": 0,
        "Individual": 1
    }

    trans_map = {
        "Automatic": 0,
        "Manual": 1
    }

    if st.button("Predict Selling Price"):

        new_car = pd.DataFrame({
            'Present_Price': [present_price],
            'Driven_kms': [driven_kms],
            'Fuel_Type': [fuel_map[fuel_type]],
            'Selling_type': [sell_map[selling_type]],
            'Transmission': [trans_map[transmission]],
            'Owner': [owner],
            'Brand': [brand],
            'Car_Age': [car_age]
        })

        prediction = rf_model.predict(new_car)[0]

        st.markdown(f"""
        <div class="pred-box">
        💰 Predicted Car Price: {prediction:.2f} Lakhs
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# KEY INSIGHTS
# ============================================================

elif section == "Key Insights":

    st.markdown('<div class="section-heading">💡 Key Insights</div>', unsafe_allow_html=True)

    most_important = pd.Series(
        rf_model.feature_importances_,
        index=X.columns
    ).idxmax()

    insights = [
        "301 cars were analyzed successfully.",
        "No missing values were found in the dataset.",
        "Random Forest performed better than Linear Regression.",
        "Present Price is the strongest predictor of selling price.",
        "Newer cars generally have higher selling prices.",
        "Cars with lower driven kilometers tend to sell at higher prices.",
        "First-owner cars have better resale value.",
        "Higher present price strongly increases predicted selling price."
    ]

    for insight in insights:
        st.markdown(f"""
        <div class="insight-box">
        ✅ {insight}
        </div>
        """, unsafe_allow_html=True)

    st.success(f"🔥 Most Important Feature: {most_important}")

