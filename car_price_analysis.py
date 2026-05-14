# ================================================
#   Car Price Prediction with Machine Learning
#   CodeAlpha Data Science Internship - Task 3
#   Student: Aiman | ID: CA/DF1/54987
# ================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------
# Matplotlib Dark Theme
# ------------------------------------------------
plt.rcParams.update({
    'figure.facecolor'  : '#0e0e14',
    'axes.facecolor'    : '#1a1730',
    'axes.edgecolor'    : '#2e2a4a',
    'axes.labelcolor'   : '#c8c4e0',
    'xtick.color'       : '#c8c4e0',
    'ytick.color'       : '#c8c4e0',
    'text.color'        : '#e8e6f0',
    'grid.color'        : '#2a2740',
    'grid.alpha'        : 0.5,
    'axes.titlecolor'   : '#e8e6f0',
    'legend.facecolor'  : '#1a1730',
    'legend.edgecolor'  : '#2e2a4a',
    'figure.edgecolor'  : '#0e0e14',
})

COLORS = ['#e879b0', '#9b59f5', '#5bbfde', '#f5a623', '#50e3c2']

print("=" * 60)
print("   Car Price Prediction - CodeAlpha Task 3")
print("=" * 60)

# ------------------------------------------------
# Step 1: Load the Dataset
# ------------------------------------------------
car = pd.read_csv('car data.csv')

# Strip spaces from column names
car.columns = car.columns.str.strip()

print("\nStep 1: Dataset loaded successfully.")
print(f"   Total Rows   : {car.shape[0]}")
print(f"   Total Columns: {car.shape[1]}")
print(f"   Columns      : {list(car.columns)}")

# ------------------------------------------------
# Step 2: Data Cleaning
# ------------------------------------------------
print("\n" + "=" * 60)
print("Step 2: Data Cleaning")
print("=" * 60)

# Keep original copy before any changes
car_original = car.copy()

print("\nFirst 5 rows of the dataset:")
print(car.head())

print("\nLast 5 rows of the dataset:")
print(car.tail())

print("\nDataset Shape:", car.shape)

print("\nChecking for Missing Values:")
print(car.isnull().sum())
if car.isnull().sum().sum() == 0:
    print("No missing values found in the dataset.")

print("\nChecking for Duplicate Rows:")
duplicates = car.duplicated().sum()
print(f"Duplicate rows found: {duplicates}")
if duplicates > 0:
    car = car.drop_duplicates()
    print(f"Duplicates removed. Rows remaining: {len(car)}")

print("\nData types:")
print(car.dtypes)

# ------------------------------------------------
# Step 3: Identify Column Types
# ------------------------------------------------
print("\n" + "=" * 60)
print("Step 3: Column Types — Numerical vs Categorical")
print("=" * 60)

numerical_cols   = car.select_dtypes(include='number').columns.tolist()
categorical_cols = car.select_dtypes(include='object').columns.tolist()

print("\nNumerical Columns:")
print(car.select_dtypes(include='number').columns.tolist())

print("\nCategorical Columns:")
print(car.select_dtypes(include='object').columns.tolist())

# ------------------------------------------------
# Step 4: Numerical Column Analysis
# ------------------------------------------------
print("\n" + "=" * 60)
print("Step 4: Numerical Columns — Descriptive Statistics")
print("=" * 60)

print("\nYear — Statistics:")
print(car['Year'].describe().round(2))

print("\nSelling Price — Statistics:")
print(car['Selling_Price'].describe().round(2))

print("\nPresent Price — Statistics:")
print(car['Present_Price'].describe().round(2))

print("\nDriven Kms — Statistics:")
print(car['Driven_kms'].describe().round(2))

print("\nOwner — Statistics:")
print(car['Owner'].describe().round(2))

print("\nVariance of Numerical Columns:")
print(car[numerical_cols].var().round(2))

print("\nStandard Deviation of Numerical Columns:")
print(car[numerical_cols].std().round(2))

print("\nCorrelation between Numerical Columns:")
print(car[numerical_cols].corr().round(2))

# ------------------------------------------------
# Step 5: Categorical Column Analysis
# ------------------------------------------------
print("\n" + "=" * 60)
print("Step 5: Categorical Columns — Value Counts")
print("=" * 60)

print("\nCar Name — Total unique values:", car['Car_Name'].nunique())
print(car['Car_Name'].value_counts().head(10))

print("\nFuel Type — Value counts:")
print(car['Fuel_Type'].value_counts())

print("\nSelling Type — Value counts:")
print(car['Selling_type'].value_counts())

print("\nTransmission — Value counts:")
print(car['Transmission'].value_counts())

# ------------------------------------------------
# Step 6: Feature Engineering
# ------------------------------------------------
print("\n" + "=" * 60)
print("Step 6: Feature Engineering")
print("=" * 60)

# Extract brand name from Car_Name (first word only)
car['Brand'] = car['Car_Name'].str.split().str[0].str.lower()

# Fix common brand name inconsistencies
car['Brand'] = car['Brand'].replace({
    'vw'       : 'volkswagen',
    'chevrolet': 'chevrolet',
    'nissan'   : 'nissan'
})

print("\nTop 10 Brands by count:")
print(car['Brand'].value_counts().head(10))

# Create Car Age feature from Year
car['Car_Age'] = 2024 - car['Year']
print(f"\nCar Age feature created (2024 - Year):")
print(car['Car_Age'].describe().round(2))

# Drop original columns that are replaced by new features
car = car.drop(['Car_Name', 'Year'], axis=1)

print("\nColumns after feature engineering:")
print(list(car.columns))

# ------------------------------------------------
# Step 7: Handle Outliers using Capping (IQR)
# ------------------------------------------------
print("\n" + "=" * 60)
print("Step 7: Handling Outliers using Capping Method")
print("=" * 60)

car_before_outliers = car.copy()
outlier_cols = ['Selling_Price', 'Present_Price', 'Driven_kms']

print(f"\nRows before capping: {len(car)}")
print("No rows will be deleted — outlier values will only be capped to bounds.")
print("\nOutlier details per column:")

for col in outlier_cols:
    Q1  = car[col].quantile(0.25)
    Q3  = car[col].quantile(0.75)
    IQR = Q3 - Q1
    lb  = Q1 - 1.5 * IQR
    ub  = Q3 + 1.5 * IQR
    outliers_count = len(car[(car[col] < lb) | (car[col] > ub)])
    car[col] = car[col].clip(lower=lb, upper=ub)
    print(f"\n   Column          : {col}")
    print(f"   Q1              : {Q1:.2f}")
    print(f"   Q3              : {Q3:.2f}")
    print(f"   IQR             : {IQR:.2f}")
    print(f"   Lower Bound     : {lb:.2f}")
    print(f"   Upper Bound     : {ub:.2f}")
    print(f"   Outliers Capped : {outliers_count}")

print(f"\nRows after capping : {len(car)}")
print(f"Rows deleted       : 0 — all rows are safe")
print("Outliers handled successfully using capping.")

# ------------------------------------------------
# Step 8: Label Encoding for Categorical Columns
# ------------------------------------------------
print("\n" + "=" * 60)
print("Step 8: Label Encoding — Categorical to Numerical")
print("=" * 60)

le = LabelEncoder()

car['Fuel_Type']    = le.fit_transform(car['Fuel_Type'])
car['Selling_type'] = le.fit_transform(car['Selling_type'])
car['Transmission'] = le.fit_transform(car['Transmission'])
car['Brand']        = le.fit_transform(car['Brand'])

print("\nEncoding complete:")
print("   Fuel_Type    : CNG=0, Diesel=1, Petrol=2")
print("   Selling_type : Dealer=0, Individual=1")
print("   Transmission : Automatic=0, Manual=1")
print("   Brand        : Numeric labels assigned")

print("\nDataset after encoding (first 5 rows):")
print(car.head())

# ------------------------------------------------
# Step 9: Full EDA
# ------------------------------------------------
print("\n" + "=" * 60)
print("Step 9: Full Data Exploration (EDA)")
print("=" * 60)

print("\nRandom 5 rows sample:")
print(car.sample(5, random_state=42))

print("\nAverage Selling Price by Fuel Type (encoded):")
print(car.groupby('Fuel_Type')['Selling_Price'].mean().round(2))

print("\nAverage Selling Price by Transmission (encoded):")
print(car.groupby('Transmission')['Selling_Price'].mean().round(2))

print("\nAverage Selling Price by Selling Type (encoded):")
print(car.groupby('Selling_type')['Selling_Price'].mean().round(2))

print("\nTop 5 most expensive cars (by Selling Price):")
print(car.nlargest(5, 'Selling_Price')[['Selling_Price', 'Present_Price', 'Car_Age']])

print("\nTop 5 cheapest cars (by Selling Price):")
print(car.nsmallest(5, 'Selling_Price')[['Selling_Price', 'Present_Price', 'Car_Age']])

# ------------------------------------------------
# Step 10: Visualizations
# ------------------------------------------------
print("\nStep 10: Generating graphs...")

# Graph 1 - Selling Price Distribution
fig1, ax1 = plt.subplots(figsize=(9, 5))
ax1.hist(car['Selling_Price'], bins=30, color='#9b59f5',
         edgecolor='#2e2a4a', alpha=0.85)
ax1.set_title('Selling Price Distribution', fontsize=14, fontweight='bold', pad=12)
ax1.set_xlabel('Selling Price (Lakhs)', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('graph1_price_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Graph 1 saved: graph1_price_distribution.png")

# Graph 2 - Correlation Heatmap
fig2, ax2 = plt.subplots(figsize=(10, 7))
corr = car.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdPu',
            square=True, linewidths=0.5, linecolor='#0e0e14',
            annot_kws={'size': 8, 'color': '#e8e6f0'}, ax=ax2)
ax2.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('graph2_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Graph 2 saved: graph2_correlation_heatmap.png")

# Graph 3 - Selling Price by Fuel Type
fig3, ax3 = plt.subplots(figsize=(8, 5))
fuel_avg = car_before_outliers.groupby('Fuel_Type')['Selling_Price'].mean().sort_values(ascending=False)
bars = ax3.bar(fuel_avg.index, fuel_avg.values,
               color=COLORS[:len(fuel_avg)], edgecolor='#2e2a4a', alpha=0.9)
for bar, val in zip(bars, fuel_avg.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.1f}L', ha='center', fontsize=10,
             color='#e8e6f0', fontweight='bold')
ax3.set_title('Average Selling Price by Fuel Type', fontsize=14, fontweight='bold', pad=12)
ax3.set_xlabel('Fuel Type', fontsize=11)
ax3.set_ylabel('Average Selling Price (Lakhs)', fontsize=11)
ax3.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('graph3_price_by_fuel.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Graph 3 saved: graph3_price_by_fuel.png")

# Graph 4 - Present Price vs Selling Price Scatter
fig4, ax4 = plt.subplots(figsize=(9, 5))
ax4.scatter(car['Present_Price'], car['Selling_Price'],
            color='#e879b0', alpha=0.6, edgecolor='#2e2a4a', s=50)
ax4.set_title('Present Price vs Selling Price', fontsize=14, fontweight='bold', pad=12)
ax4.set_xlabel('Present Price (Lakhs)', fontsize=11)
ax4.set_ylabel('Selling Price (Lakhs)', fontsize=11)
ax4.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('graph4_present_vs_selling.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Graph 4 saved: graph4_present_vs_selling.png")

# Graph 5 - Driven KMs vs Selling Price
fig5, ax5 = plt.subplots(figsize=(9, 5))
ax5.scatter(car['Driven_kms'], car['Selling_Price'],
            color='#5bbfde', alpha=0.6, edgecolor='#2e2a4a', s=50)
ax5.set_title('Driven KMs vs Selling Price', fontsize=14, fontweight='bold', pad=12)
ax5.set_xlabel('Driven KMs', fontsize=11)
ax5.set_ylabel('Selling Price (Lakhs)', fontsize=11)
ax5.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('graph5_driven_vs_selling.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Graph 5 saved: graph5_driven_vs_selling.png")

# Graph 6 - Car Age vs Selling Price
fig6, ax6 = plt.subplots(figsize=(9, 5))
ax6.scatter(car['Car_Age'], car['Selling_Price'],
            color='#f5a623', alpha=0.6, edgecolor='#2e2a4a', s=50)
ax6.set_title('Car Age vs Selling Price', fontsize=14, fontweight='bold', pad=12)
ax6.set_xlabel('Car Age (Years)', fontsize=11)
ax6.set_ylabel('Selling Price (Lakhs)', fontsize=11)
ax6.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('graph6_age_vs_selling.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Graph 6 saved: graph6_age_vs_selling.png")

# ------------------------------------------------
# Step 11: Prepare Data for Model
# ------------------------------------------------
print("\n" + "=" * 60)
print("Step 11: Data Preparation for Model")
print("=" * 60)

# Features and Target
X = car.drop('Selling_Price', axis=1)
y = car['Selling_Price']

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target  (y) shape: {y.shape}")
print(f"\nFeature columns: {list(X.columns)}")

# Split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nData split complete.")
print(f"   Training samples : {X_train.shape[0]}")
print(f"   Testing samples  : {X_test.shape[0]}")

# ------------------------------------------------
# Step 12: Train Linear Regression Model
# ------------------------------------------------
print("\n" + "=" * 60)
print("Step 12: Model Training — Linear Regression")
print("=" * 60)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr  = lr_model.predict(X_test)
r2_lr      = r2_score(y_test, y_pred_lr)
mae_lr     = mean_absolute_error(y_test, y_pred_lr)
mse_lr     = mean_squared_error(y_test, y_pred_lr)
rmse_lr    = np.sqrt(mse_lr)

print("\nLinear Regression Results:")
print(f"   R2 Score : {r2_lr:.4f}  ({r2_lr*100:.2f}% variance explained)")
print(f"   MAE      : {mae_lr:.4f}  (avg error in Lakhs)")
print(f"   MSE      : {mse_lr:.4f}")
print(f"   RMSE     : {rmse_lr:.4f}")

# ------------------------------------------------
# Step 13: Train Random Forest Regressor
# ------------------------------------------------
print("\n" + "=" * 60)
print("Step 13: Model Training — Random Forest Regressor")
print("=" * 60)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf  = rf_model.predict(X_test)
r2_rf      = r2_score(y_test, y_pred_rf)
mae_rf     = mean_absolute_error(y_test, y_pred_rf)
mse_rf     = mean_squared_error(y_test, y_pred_rf)
rmse_rf    = np.sqrt(mse_rf)

print("\nRandom Forest Results:")
print(f"   R2 Score : {r2_rf:.4f}  ({r2_rf*100:.2f}% variance explained)")
print(f"   MAE      : {mae_rf:.4f}  (avg error in Lakhs)")
print(f"   MSE      : {mse_rf:.4f}")
print(f"   RMSE     : {rmse_rf:.4f}")

# ------------------------------------------------
# Step 14: Model Comparison
# ------------------------------------------------
print("\n" + "=" * 60)
print("Step 14: Model Comparison")
print("=" * 60)

print(f"\n{'Model':<25} {'R2 Score':<12} {'MAE':<10} {'RMSE':<10}")
print("-" * 60)
print(f"{'Linear Regression':<25} {r2_lr:<12.4f} {mae_lr:<10.4f} {rmse_lr:<10.4f}")
print(f"{'Random Forest':<25} {r2_rf:<12.4f} {mae_rf:<10.4f} {rmse_rf:<10.4f}")

best_model = "Random Forest" if r2_rf > r2_lr else "Linear Regression"
print(f"\nBest Model: {best_model} (higher R2 Score is better)")

# ------------------------------------------------
# Step 15: Graph 7 - Feature Importance
# ------------------------------------------------
importances   = rf_model.feature_importances_
feature_names = list(X.columns)
feat_imp      = pd.Series(importances, index=feature_names).sort_values()

fig7, ax7 = plt.subplots(figsize=(9, 6))
bar_colors = ['#e879b0' if v == feat_imp.max() else '#9b59f5'
              for v in feat_imp.values]
bars = ax7.barh(feat_imp.index, feat_imp.values,
                color=bar_colors, edgecolor='#2e2a4a')
for bar, val in zip(bars, feat_imp.values):
    ax7.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=9,
             color='#e8e6f0', fontweight='bold')
ax7.set_title('Feature Importance — Random Forest', fontsize=14, fontweight='bold', pad=12)
ax7.set_xlabel('Importance Score', fontsize=11)
ax7.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('graph7_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Graph 7 saved: graph7_feature_importance.png")

# Graph 8 - Actual vs Predicted (Random Forest)
fig8, ax8 = plt.subplots(figsize=(8, 6))
ax8.scatter(y_test, y_pred_rf, color='#9b59f5',
            alpha=0.7, edgecolor='#2e2a4a', s=60)
min_val = min(y_test.min(), y_pred_rf.min())
max_val = max(y_test.max(), y_pred_rf.max())
ax8.plot([min_val, max_val], [min_val, max_val],
         color='#e879b0', linewidth=2, linestyle='--', label='Perfect Prediction')
ax8.set_title('Actual vs Predicted Selling Price (Random Forest)',
              fontsize=13, fontweight='bold', pad=12)
ax8.set_xlabel('Actual Price (Lakhs)', fontsize=11)
ax8.set_ylabel('Predicted Price (Lakhs)', fontsize=11)
ax8.legend(fontsize=10)
ax8.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('graph8_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Graph 8 saved: graph8_actual_vs_predicted.png")

# ------------------------------------------------
# Step 16: Predict a New Car Price
# ------------------------------------------------
print("\n" + "=" * 60)
print("Step 16: Predicting Price of a New Car")
print("=" * 60)

# Example: A 5-year-old Petrol, Dealer, Manual car
# Petrol=2, Dealer=0, Manual=1, Brand=encoded value
new_car = pd.DataFrame({
    'Selling_Price'  : [0],       # placeholder — will be predicted
    'Present_Price'  : [7.5],
    'Driven_kms'     : [40000],
    'Fuel_Type'      : [2],       # Petrol
    'Selling_type'   : [0],       # Dealer
    'Transmission'   : [1],       # Manual
    'Owner'          : [0],       # First owner
    'Brand'          : [10],      # Encoded brand
    'Car_Age'        : [5],
})
new_car = new_car.drop('Selling_Price', axis=1)

predicted_price = rf_model.predict(new_car)[0]

print("\nNew Car Details:")
print(f"   Present Price  : 7.5 Lakhs")
print(f"   Driven KMs     : 40,000")
print(f"   Fuel Type      : Petrol")
print(f"   Selling Type   : Dealer")
print(f"   Transmission   : Manual")
print(f"   Owner          : First Owner")
print(f"   Car Age        : 5 Years")
print(f"\nPredicted Selling Price: {predicted_price:.2f} Lakhs")

# ------------------------------------------------
# Step 17: Key Insights
# ------------------------------------------------
print("\n" + "=" * 60)
print("Step 17: Key Insights")
print("=" * 60)

most_important = feat_imp.idxmax()

print(f"""
1. DATASET:
   - 301 cars analyzed with 9 features.
   - No missing values found in the dataset.
   - Selling price ranges from 0.1 to 35 Lakhs.

2. FEATURE ENGINEERING:
   - Brand extracted from Car_Name column.
   - Car_Age created from Year column.
   - All categorical columns label encoded.

3. OUTLIER HANDLING:
   - Capping method used — 0 rows deleted.
   - Extreme Driven_kms and Present_Price values capped.

4. MODEL PERFORMANCE:
   - Linear Regression R2  : {r2_lr*100:.2f}%
   - Random Forest R2      : {r2_rf*100:.2f}%
   - Best Model            : {best_model}

5. MOST IMPORTANT FEATURE:
   - {most_important} is the strongest predictor of car selling price.

6. KEY FINDINGS:
   - Newer cars (lower Car_Age) sell for higher prices.
   - Diesel cars have higher average selling price than Petrol.
   - First owner cars sell for significantly more than multi-owner cars.
   - Higher Present_Price strongly correlates with higher Selling_Price.
""")

# ------------------------------------------------
# Final Summary
# ------------------------------------------------
print("=" * 60)
print("   Project Complete!")
print("=" * 60)
print(f"Dataset        : car_data.csv (301 rows, 9 columns)")
print(f"Outliers       : Handled using Capping Method")
print(f"Models Trained : Linear Regression + Random Forest")
print(f"Best Model     : {best_model}")
print(f"Best R2 Score  : {max(r2_lr, r2_rf)*100:.2f}%")
print(f"Graphs Saved   : 8 PNG files")
print("=" * 60)