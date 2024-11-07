import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('diabetes_data.csv')

# Ensure 'Outcome' is binary
data['Outcome'] = data['Outcome'].astype(int)

# Improved synthetic sample generation with more realistic constraints
def generate_synthetic_sample(data):
    synthetic_sample = {}
    
    # Add realistic constraints for each feature
    synthetic_sample['Pregnancies'] = np.random.randint(0, 15)
    synthetic_sample['Glucose'] = np.random.randint(70, 200)
    synthetic_sample['BloodPressure'] = np.random.randint(60, 130)
    synthetic_sample['SkinThickness'] = np.random.randint(20, 50)
    synthetic_sample['Insulin'] = np.random.randint(15, 276)
    synthetic_sample['BMI'] = np.random.uniform(18.5, 40)
    synthetic_sample['DiabetesPedigreeFunction'] = np.random.uniform(0.1, 2.5)
    synthetic_sample['Age'] = np.random.randint(21, 70)
    
    # More sophisticated outcome determination based on medical criteria
    risk_score = (
        (synthetic_sample['Glucose'] > 140) * 2.5 +
        (synthetic_sample['BMI'] > 30) * 1.8 +
        (synthetic_sample['Age'] > 45) * 1.2 +
        (synthetic_sample['BloodPressure'] > 90) * 1.3 +
        (synthetic_sample['DiabetesPedigreeFunction'] > 0.8) * 1.5
    )
    synthetic_sample['Outcome'] = 1 if (risk_score > 3.0 or np.random.random() < 0.25) else 0
    
    return synthetic_sample

# Generate 200 new samples (increased from 100)
new_samples = [generate_synthetic_sample(data) for _ in range(200)]
synthetic_data = pd.DataFrame(new_samples)

# Combine with original data
data = pd.concat([data, synthetic_data], ignore_index=True)

# Replace 0 values with NaN for certain features
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in zero_not_accepted:
    data[column] = data[column].replace(0, np.NaN)

# Impute missing values using KNNImputer with more neighbors
imputer = KNNImputer(n_neighbors=7)
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Enhanced feature engineering
data_imputed['GlucoseToInsulinRatio'] = data_imputed['Glucose'] / data_imputed['Insulin']
data_imputed['BMI_Category'] = pd.cut(data_imputed['BMI'], 
                                    bins=[0, 18.5, 25, 30, 35, 100], 
                                    labels=[0, 1, 2, 3, 4])
data_imputed['Age_Category'] = pd.cut(data_imputed['Age'], 
                                    bins=[0, 25, 35, 45, 55, 100], 
                                    labels=[0, 1, 2, 3, 4])
data_imputed['GlucoseBMI'] = data_imputed['Glucose'] * data_imputed['BMI']
data_imputed['IsHypertensive'] = (data_imputed['BloodPressure'] > 90).astype(int)
data_imputed['IsObese'] = (data_imputed['BMI'] >= 30).astype(int)
data_imputed['HighGlucose'] = (data_imputed['Glucose'] >= 140).astype(int)
data_imputed['MetabolicScore'] = (data_imputed['BMI'] * data_imputed['Glucose'] * 
                                 data_imputed['DiabetesPedigreeFunction']) / 1000
data_imputed['InsulinBMIInteraction'] = data_imputed['Insulin'] * data_imputed['BMI']

# Create polynomial features for key indicators
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = ['Glucose', 'BMI', 'DiabetesPedigreeFunction']
poly_data = poly.fit_transform(data_imputed[poly_features])
poly_feature_names = [f'Poly_{i}' for i in range(poly_data.shape[1])]
data_imputed = pd.concat([data_imputed, 
                         pd.DataFrame(poly_data[:, len(poly_features):], 
                                    columns=poly_feature_names[len(poly_features):])], 
                        axis=1)

# Separate features and target
X = data_imputed.drop('Outcome', axis=1)
y = data_imputed['Outcome'].astype(int)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=y)

# Enhanced SMOTE with better sampling strategy
smote = SMOTE(random_state=42, k_neighbors=7)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Improved Random Forest model with better hyperparameters
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced_subsample',
    random_state=42,
    bootstrap=True,
    n_jobs=-1
)

# Train model
model.fit(X_train_resampled, y_train_resampled)

# Cross-validation
cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())

# Evaluate on test set
y_pred = model.predict(X_test)
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Feature Importance
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
plt.title('Top 15 Important Features')
plt.tight_layout()
plt.show()

print("\nTop 10 Important Features:")
print(feature_importance.head(10))