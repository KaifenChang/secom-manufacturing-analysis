import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
from xgboost import XGBClassifier 

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# Data processing
## read data files
data_file = "secom/secom.data"
df_data = pd.read_csv(data_file, sep="\s+", header=None)

label_file = "secom/secom_labels.data"
df_label = pd.read_csv(label_file, sep="\s+", header=None, names=["Label", "Timestamp"])

print(df_data.head())
print(df_label.head())

## merge data and label
df_combined = pd.concat([df_data, df_label], axis=1)

## assign column names
columns = ["Feature" + str(i) for i in range(df_data.shape[1])]
columns += ["Label", "Timestamp"]
df_combined.columns = columns

## save to csv
df_combined.to_csv("secom_combined.csv", index=False)
print("Saved to secom_combined.csv")

# Exploratory data analysis
## check missing values
df_combined = pd.read_csv("secom_combined.csv")

missing_values = df_combined.isnull().sum()
missing_percentage = (missing_values / len(df_combined)) * 100

missing_df = pd.DataFrame({"Missing Count": missing_values, "Missing %": missing_percentage})
missing_df = missing_df[missing_df["Missing Count"] > 0]  

print("Columns with missing values:")
print(missing_df.sort_values(by="Missing %", ascending=False))

## handle missing values
## drop columns with more than 50% missing values
missing_threshold = 50
columns_to_drop = missing_df[missing_df["Missing %"] > missing_threshold].index
df_cleaned = df_combined.drop(columns=columns_to_drop)

print(f"Dropped columns with more than {missing_threshold}% missing values")

## imputing missing values (use median)
numeric_cols = df_cleaned.select_dtypes(include=["number"]).columns
df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(
    df_cleaned[numeric_cols].median())

print("Imputed missing values with median")

## save cleaned data
df_cleaned.to_csv("secom_cleaned.csv", index=False)
print("Saved to secom_cleaned.csv")

## check class distribution (outlier and skewness)
## check outliner using IQR
Q1 = df_cleaned[numeric_cols].quantile(0.25)
Q3 = df_cleaned[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


print(f"df_cleaned[numeric_cols].shape: {df_cleaned[numeric_cols].shape}")  # 檢查 DataFrame 大小
print(f"Q1.shape: {Q1.shape}, Q3.shape: {Q3.shape}, IQR.shape: {IQR.shape}")  # 檢查 IQR 的大小


# 設定 IQR 過濾條件（確保數據對齊）
filter_condition = ~((df_cleaned[numeric_cols] < (Q1 - 1.5 * IQR)) | (df_cleaned[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)

# 允許 `Label=1` 的數據保留
df_filtered = df_cleaned[(filter_condition) | (df_cleaned["Label"] == -1) | (df_cleaned["Label"] == 1)]

# 檢查 `Label` 分佈
print(df_filtered["Label"].value_counts())

# 儲存修正後的數據
df_filtered.to_csv("secom_IQR_filtered.csv", index=False)
print("Saved to secom_IQR_filtered.csv")


sample_features = df_filtered.iloc[:, :10]

plt.figure(figsize=(12, 6))
sns.boxplot(data=sample_features)
plt.xticks(rotation=90)
plt.title("Boxplot of Features (Checking Outliers)")
#plt.show()

# check outlier
outlier_counts = (
    (df_cleaned[numeric_cols] < lower_bound) | (df_cleaned[numeric_cols] > upper_bound)
).sum()

outlier_counts = outlier_counts.sort_values(ascending=False)

print("Outlier detection completed! Top features with most outliers:")
print(outlier_counts.head(20))

plt.figure(figsize=(12, 6))
outlier_counts.head(20).plot(kind="bar")
plt.title("Number of Outliers per Feature (Top 20)")
plt.xlabel("Features")
plt.ylabel("Outlier Count")
plt.xticks(rotation=90)
#plt.show()


# capping outliers
df_capped = df_filtered.copy()
df_capped[numeric_cols] = df_capped[numeric_cols].clip(lower=lower_bound, upper=upper_bound, axis=1)

# Ensure Label is preserved
df_capped['Label'] = df_filtered['Label']

print("Capped outliers using IQR")

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_capped.iloc[:, :10])
plt.xticks(rotation=90)
plt.title("Boxplot After Capping Outliers")
# plt.show()

plt.figure(figsize=(12, 6))
for feature in df_capped.columns[:5]:  # Check first 5 features
    sns.histplot(df_capped[feature], kde=True, label=feature, alpha=0.5)
plt.legend()
plt.title("Feature Distribution After Capping")
# plt.show()

df_capped.to_csv("secom_capped.csv", index=False)
print("Saved to secom_capped.csv")


## check the skewness of the data
skewness = df_capped[numeric_cols].skew().sort_values(ascending=False)
print("Skewness of the features:")
print(skewness.head(20))

# 選擇高度偏斜的特徵
highly_skewed_features = skewness[abs(skewness) > 1].index

df_transformed = df_capped.copy()
for feature in highly_skewed_features:
    # Ensure all data is positive by adding a constant
    min_value = df_transformed[feature].min()
    if min_value <= 0:
        df_transformed[feature] += abs(min_value) + 1  # Shift data to be positive
    df_transformed[feature], _ = boxcox(df_transformed[feature])

# Ensure Label is preserved
df_transformed['Label'] = df_capped['Label']

# 重新畫圖
plt.figure(figsize=(12, 6))
for feature in df_transformed.columns[:5]:  
    sns.histplot(df_transformed[feature], kde=True, label=feature, alpha=0.5)
plt.legend()
plt.title("Feature Distribution After Skewness Transformation")
#plt.show()

skewness_after = df_transformed[numeric_cols].skew().sort_values(ascending=False)
print("Features with highest skewness after transformation:")
print(skewness_after.head(10))

## save to csv
df_transformed.to_csv("secom_processed.csv", index=False)
print("Saved to secom_processed.csv")


# correlation analysis
## Calculate correlation matrix excluding non-numeric columns
correlation_matrix = df_transformed[numeric_cols].corr()

# plot correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", center=0)
plt.title("Correlation Matrix of Features")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
# plt.show()

print("Correlation matrix plotted successfully")

# Handle multicollinearity
# Set a threshold for identifying multicollinearity
correlation_threshold = 0.8

# Get the upper triangle of the correlation matrix
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than the threshold
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)]

# Optionally, you can print the pairs of features that are highly correlated
print("Highly correlated feature pairs:")
for column in to_drop:
    correlated_features = upper_triangle.index[upper_triangle[column] > correlation_threshold].tolist()
    for feature in correlated_features:
        print(f"{column} - {feature}: {correlation_matrix.loc[column, feature]:.2f}")

# Drop the features identified
df_reduced = df_transformed.drop(columns=to_drop)

print(f"Removed {len(to_drop)} highly collinear features")
print("Remaining features:", len(df_reduced.columns))

# Save the reduced dataset
df_reduced.to_csv("secom_reduced.csv", index=False)
print("Saved reduced dataset to secom_reduced.csv")

# Compute feature-label correlation
feature_label_corr = correlation_matrix["Label"].abs().sort_values(ascending=False)
print("Top features correlated with Label:\n", feature_label_corr.head(10))

# plot feature-label correlation
plt.figure(figsize=(12, 6))
feature_label_corr.plot(kind="bar")
plt.title("Feature-Label Correlation")
plt.xlabel("Features")
plt.ylabel("Correlation")
plt.xticks(rotation=90)
# plt.show()

# Feature importance using XGBoost
print("\nCalculating feature importance using XGBoost...")

# Prepare the data
X = df_reduced.drop(['Label', 'Timestamp'], axis=1)
y = df_reduced['Label']

# Convert labels from -1, 1 to 0, 1
y = y.replace(-1, 0)

# Updated XGBoost initialization with hyperparameter tuning and regularization
xgb = XGBClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=4,       
    learning_rate=0.1, 
    reg_alpha=0.1,     # L1 regularization; encourages sparsity
    reg_lambda=1.0,    # L2 regularization; prevents overfitting
    gamma=0,
    importance_type='weight'  # Options include: 'weight', 'gain', 'cover'
)
xgb.fit(X, y)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb.feature_importances_
})

# Sort by importance
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Save feature importance to CSV
feature_importance.to_csv('feature_importance.csv', index=False)
print("Feature importance saved to feature_importance.csv")

# Plot feature importance
plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_importance)), feature_importance['importance'])
plt.xticks(range(len(feature_importance)), feature_importance['feature'], rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
#plt.show()

print("Feature importance plot saved to feature_importance.png")

# Determine a threshold for feature importance
importance_threshold = 0.01  # Example threshold, adjust based on your needs

# Identify features to drop
features_to_drop = feature_importance[feature_importance['importance'] < importance_threshold]['feature'].tolist()

# Drop the non-important features
df_final = df_reduced.drop(columns=features_to_drop)

print(f"Dropped {len(features_to_drop)} non-important features")
print("Remaining features:", len(df_final.columns))

# Save the final dataset
df_final.to_csv("secom_final.csv", index=False)
print("Saved final dataset to secom_final.csv")

# Update numeric_cols to reflect the columns in df_final
numeric_cols = df_final.select_dtypes(include=["number"]).columns

# Plot feature correlation heatmap with the updated numeric_cols
plt.figure(figsize=(10, 8))
sns.heatmap(df_final[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

