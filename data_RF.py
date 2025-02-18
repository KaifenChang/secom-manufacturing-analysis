import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE


# 1_Data processing

# read data files
data_file = "secom/secom.data"
df_data = pd.read_csv(data_file, sep="\s+", header=None)

label_file = "secom/secom_labels.data"
df_label = pd.read_csv(label_file, sep="\s+", header=None, names=["Label", "Timestamp"])

print(df_data.head())
print(df_label.head())

# merge data and label
df_combined = pd.concat([df_data, df_label], axis=1)

# assign column names
columns = ["Feature" + str(i) for i in range(df_data.shape[1])]
columns += ["Label", "Timestamp"]
df_combined.columns = columns

# save to csv
df_combined.to_csv("secom_combined.csv", index=False)
print("Saved to secom_combined.csv")

# 2_Exploratory data analysis

# 2_1_Handling missing values
# check missing values
df_combined = pd.read_csv("secom_combined.csv")

missing_values = df_combined.isnull().sum()
missing_percentage = (missing_values / len(df_combined)) * 100

missing_df = pd.DataFrame({"Missing Count": missing_values, "Missing %": missing_percentage})
missing_df = missing_df[missing_df["Missing Count"] > 0]  

print("Columns with missing values:")
print(missing_df.sort_values(by="Missing %", ascending=False))

# handle missing values
# drop columns with more than 50% missing values
missing_threshold = 50
columns_to_drop = missing_df[missing_df["Missing %"] > missing_threshold].index
df_cleaned = df_combined.drop(columns=columns_to_drop)

print(f"Dropped columns with more than {missing_threshold}% missing values")

# imputing missing values (use median)
numeric_cols = df_cleaned.select_dtypes(include=["number"]).columns
df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(
    df_cleaned[numeric_cols].median())

print("Imputed missing values with median")

# save cleaned data
df_cleaned.to_csv("secom_cleaned.csv", index=False)
print("Saved to secom_cleaned.csv")

# 2_3_Outlier detection using IQR
# check class distribution (outlier and skewness)
# check outliner using IQR
Q1 = df_cleaned[numeric_cols].quantile(0.25)
Q3 = df_cleaned[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"df_cleaned[numeric_cols].shape: {df_cleaned[numeric_cols].shape}")  # 檢查 DataFrame 大小
print(f"Q1.shape: {Q1.shape}, Q3.shape: {Q3.shape}, IQR.shape: {IQR.shape}")  # 檢查 IQR 的大小


# set IQR filter condition
filter_condition = ~((df_cleaned[numeric_cols] < (Q1 - 1.5 * IQR)) | (df_cleaned[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)

# allow `Label=1` data to be retained
df_filtered = df_cleaned[(filter_condition) | (df_cleaned["Label"] == -1) | (df_cleaned["Label"] == 1)]

# check `Label` distribution
print(df_filtered["Label"].value_counts())

# save the corrected data
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


# 2_4 check the skewness of the data
skewness = df_capped[numeric_cols].skew().sort_values(ascending=False)
print("Skewness of the features:")
print(skewness.head(20))

# 2_5_Skewness transformation
# select highly skewed features
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

# plot the distribution of the transformed features
plt.figure(figsize=(12, 6))
for feature in df_transformed.columns[:5]:  
    sns.histplot(df_transformed[feature], kde=True, label=feature, alpha=0.5)
plt.legend()
plt.title("Feature Distribution After Skewness Transformation")
#plt.show()

skewness_after = df_transformed[numeric_cols].skew().sort_values(ascending=False)
print("Features with highest skewness after transformation:")
print(skewness_after.head(10))

# save to csv
df_transformed.to_csv("secom_processed.csv", index=False)
print("Saved to secom_processed.csv")


# 3_correlation analysis
# 3_1 Calculate correlation matrix excluding non-numeric columns
correlation_matrix = df_transformed[numeric_cols].corr()

# plot correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", center=0)
plt.title("Correlation Matrix of Features")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
# plt.show()

print("Correlation matrix plotted successfully")

# 3_2 Handle multicollinearity
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

# 3_3 plot feature-label correlation
plt.figure(figsize=(12, 6))
feature_label_corr.plot(kind="bar")
plt.title("Feature-Label Correlation")
plt.xlabel("Features")
plt.ylabel("Correlation")
plt.xticks(rotation=90)
# plt.show()

# 4_Feature importance using Random Forest

print("\nCalculating feature importance using Random Forest...")
# 4_1 Prepare and train the data
X = df_reduced.drop(['Label', 'Timestamp'], axis=1)
y = df_reduced['Label']

# Initialize and train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)    

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_,
    'std': np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
})  

# Sort by importance
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Save feature importance to CSV
feature_importance.to_csv('feature_importance.csv', index=False)
print("Feature importance saved to feature_importance.csv") 

# plot feature importance
plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_importance)), feature_importance['importance'])
plt.xticks(range(len(feature_importance)), feature_importance['feature'], rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importance')   
plt.tight_layout()
plt.savefig('feature_importance.png')
# plt.show()

print("Feature importance plot saved to feature_importance.png")

# 4_2 Use feature importance to select features
# Determine a threshold for feature importance
importance_threshold = 0.01  # Example threshold, adjust based on your needs

# Identify features to drop
features_to_drop = feature_importance[feature_importance['importance'] < importance_threshold]['feature'].tolist()

# Drop the non-important features
df_final= df_reduced.drop(columns=features_to_drop)

print(f"Dropped {len(features_to_drop)} non-important features")
print("Remaining features:", len(df_final.columns))

# Save the final dataset
df_final.to_csv("secom_final.csv", index=False)
print("Saved final dataset to secom_final.csv")

# Update numeric_cols to reflect the columns in df_final
numeric_cols = df_final.select_dtypes(include=["number"]).columns.drop('Label')

# 4_3 Plot feature correlation heatmap with the updated numeric_cols
plt.figure(figsize=(10, 8))
sns.heatmap(df_final[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
#plt.show()

# 5_Train the model
# 5_1 Data Preparation
X = df_final[numeric_cols]
y = df_final['Label']

# Convert labels from -1, 1 to 0, 1 for binary classification
y = (y > 0).astype(int)

# Print data shapes and check for nulls
print("\nData Overview:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print("\nNull values in X:", X.isnull().sum().sum())
print("Null values in y:", y.isnull().sum())
print("\nClass distribution:", np.bincount(y))

# 5_2 Split Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Step 2.5: Apply SMOTE to handle class imbalance
print("\nApplying SMOTE to balance the training data...")
print("Before SMOTE:", np.bincount(y_train))
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("After SMOTE:", np.bincount(y_train_resampled))

# 5_3 Train Models
print("\nTraining models...")

# Train Random Forest
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1  # Use all available cores
)

# Train XGBoost
scale_pos_weight = len(y[y == 0]) / len(y[y == 1])
xgb_classifier = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='auc'
)

# Fit both models
print("Fitting Random Forest...")
rf_classifier.fit(X_train_resampled, y_train_resampled)

print("Fitting XGBoost...")
xgb_classifier.fit(X_train_resampled, y_train_resampled)

# 5_4 Evaluate both models
print("\nRandom Forest Results:")
rf_train_pred = rf_classifier.predict(X_train_resampled)
rf_test_pred = rf_classifier.predict(X_test)
print("\nTraining Accuracy:", accuracy_score(y_train_resampled, rf_train_pred))
print("Test Accuracy:", accuracy_score(y_test, rf_test_pred))
print("\nClassification Report (Test Set):")
print(classification_report(y_test, rf_test_pred))

print("\nXGBoost Results:")
xgb_train_pred = xgb_classifier.predict(X_train_resampled)
xgb_test_pred = xgb_classifier.predict(X_test)
print("\nTraining Accuracy:", accuracy_score(y_train_resampled, xgb_train_pred))
print("Test Accuracy:", accuracy_score(y_test, xgb_test_pred))
print("\nClassification Report (Test Set):")
print(classification_report(y_test, xgb_test_pred))

# 5_5 Confusion Matrix Visualization
plt.figure(figsize=(12, 5))

# Training set confusion matrix
plt.subplot(1, 2, 1)
conf_matrix_train = confusion_matrix(y_train_resampled, rf_train_pred)
sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues')
plt.title('Training Set Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')

# Test set confusion matrix
plt.subplot(1, 2, 2)
conf_matrix_test = confusion_matrix(y_test, rf_test_pred)
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues')
plt.title('Test Set Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')

plt.tight_layout()
plt.show()

# Save model evaluation results
evaluation_results = {
    "Training Accuracy": accuracy_score(y_train_resampled, rf_train_pred),
    "Testing Accuracy": accuracy_score(y_test, rf_test_pred),
    "Training Classification Report": classification_report(y_train_resampled, rf_train_pred),
    "Testing Classification Report": classification_report(y_test, rf_test_pred)
}

# Write evaluation results to file
with open('model_evaluation_results.txt', 'w') as f:
    f.write("Random Forest Model Evaluation Results\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Training Set Accuracy: {evaluation_results['Training Accuracy']:.4f}\n")
    f.write(f"Test Set Accuracy: {evaluation_results['Testing Accuracy']:.4f}\n\n")
    f.write("Training Set Classification Report:\n")
    f.write(evaluation_results['Training Classification Report'])
    f.write("\nTest Set Classification Report:\n")
    f.write(evaluation_results['Testing Classification Report'])

print("\nModel evaluation results have been saved to 'model_evaluation_results.txt'")
