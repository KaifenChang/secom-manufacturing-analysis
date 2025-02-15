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

# Load the transformed dataset
df_cleaned_no_collinear = pd.read_csv("secom_transformed.csv")

# Separate features and label
features = [col for col in df_cleaned_no_collinear.columns if col not in ['Label', 'Timestamp']]
X = df_cleaned_no_collinear[features]
y = df_cleaned_no_collinear['Label']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Print some information about our data
print("\nData Information:")
print(f"Number of samples: {len(X)}")
print(f"Number of features: {len(features)}")
print(f"Label distribution:\n{y.value_counts()}")
print("\nFeature value ranges:")
print(pd.DataFrame(X_scaled).describe())

# Train Random Forest with different parameters
rf_model = RandomForestClassifier(
    n_estimators=1000,          # More trees
    max_depth=None,             # Allow full depth
    min_samples_split=2,        # Default split criterion
    min_samples_leaf=1,         # Default leaf size
    max_features='sqrt',        # Traditional RF feature selection
    class_weight='balanced',    # Handle class imbalance
    random_state=42
)
rf_model.fit(X_scaled, y)

# Calculate feature importance using permutation importance
perm_importance = permutation_importance(
    rf_model, X_scaled, y,
    n_repeats=10,
    random_state=42
)

# Create feature importance DataFrame
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': perm_importance.importances_mean,
    'std': perm_importance.importances_std
})

# Sort features by importance
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Get top 10 features
top_10_features = feature_importance.head(10)

# Create bar plot of feature importance
plt.figure(figsize=(12, 6))
sns.barplot(
    data=top_10_features,
    x='feature',
    y='importance',
    color='skyblue',
    yerr=top_10_features['std']  # Add error bars
)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Permutation Importance Score')
plt.title('Top 10 Most Important Features (Random Forest with Permutation Importance)')

# Add value labels on top of each bar
for i, v in enumerate(top_10_features['importance']):
    plt.text(i, v, f'{v:.4f}', ha='center', va='bottom')

# Adjust y-axis to start from 0 and add some padding at the top
max_importance = top_10_features['importance'].max()
plt.ylim(0, max_importance * 1.2)

plt.tight_layout()
plt.show()

print("\nTop 10 most important features:")
print(top_10_features[['feature', 'importance']].to_string())

# Print some additional statistics
print("\nFeature Importance Statistics:")
print(f"Max importance: {feature_importance['importance'].max():.6f}")
print(f"Mean importance: {feature_importance['importance'].mean():.6f}")
print(f"Number of features with importance > 0: {(feature_importance['importance'] > 0).sum()}")

# Save feature importance results
feature_importance.to_csv("feature_importance.csv", index=False)
print("\nFeature importance saved to feature_importance.csv")