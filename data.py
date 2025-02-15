import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import yeojohnson
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

outlier_counts = (
    (df_cleaned[numeric_cols] < lower_bound) | (df_cleaned[numeric_cols] > upper_bound)
).sum()

outlier_counts = outlier_counts.sort_values(ascending=False)

print("Outlier detection completed! Top features with most outliers:")
print(outlier_counts.head(20))

df_capped = df_cleaned.copy()
df_capped[numeric_cols] = df_capped[numeric_cols].clip(lower=lower_bound, upper=upper_bound, axis=1)

print("Capped outliers using IQR")

## check the skewness of the data
skewness = df_capped[numeric_cols].skew().sort_values(ascending=False)
print("Skewness of the features:")
print(skewness.head(20))

df_transformed = df_capped.copy()
for feature in skewness[abs(skewness) > 1].index:
    if (df_transformed[feature] > 0).all():
        if skewness[feature] > 1.5:
            df_transformed[feature] = np.log1p(df_transformed[feature])
        else:
            df_transformed[feature] = np.sqrt(df_transformed[feature])
    else:
        df_transformed[feature], _ = yeojohnson(df_transformed[feature])

print("Transformed highly skewed features")

## save transformed data
df_transformed.to_csv("secom_transformed.csv", index=False)

print("Saved to secom_transformed.csv")

# correlation analysis
## Ensure Timestamp is not included in numeric columns
numeric_cols = df_transformed.select_dtypes(include=["number"]).columns

## Calculate correlation matrix excluding non-numeric columns
correlation_matrix = df_transformed[numeric_cols].corr()

## Plot correlation matrix
plt.figure(figsize=(12, 8))

## Use a more efficient plotting method if needed
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", center=0)
plt.title("Correlation Matrix of Features")
plt.tight_layout()
plt.show()

print("Correlation matrix plotted successfully")
