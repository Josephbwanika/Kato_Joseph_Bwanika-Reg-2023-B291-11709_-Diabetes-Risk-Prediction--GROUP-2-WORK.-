# =============================================================================
# DIABETES RISK PREDICTION - CONCISE IMPLEMENTATION
# PIMA Indians Diabetes Dataset Analysis
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy.stats as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =============================================================================
# PART A: DATA LOADING & PREPROCESSING
# =============================================================================

print("PART A: DATA LOADING & PREPROCESSING")
print("=" * 50)

# Load dataset
df = pd.read_csv('diabetes.csv')
print(f"Dataset Shape: {df.shape}")

# Handle invalid zero values
columns_with_invalid_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df_original = df.copy()

for column in columns_with_invalid_zeros:
    df[column] = df[column].replace(0, np.nan)
    df[column].fillna(df[column].median(), inplace=True)

# Normalize continuous variables
continuous_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
scaler = StandardScaler()
df[continuous_columns] = scaler.fit_transform(df[continuous_columns])

# Handle outliers
Q1 = df[continuous_columns].quantile(0.25)
Q3 = df[continuous_columns].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

for col in continuous_columns:
    df[col] = np.where(df[col] < lower_bound[col], lower_bound[col], df[col])
    df[col] = np.where(df[col] > upper_bound[col], upper_bound[col], df[col])

# =============================================================================
# PART B: FIRST EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

print("\nPART B: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 50)

# Descriptive statistics
key_features = ['Glucose', 'BMI', 'Age']
print("Descriptive Statistics:")
print(df[key_features].describe())

# Histograms for continuous features
plt.figure(figsize=(15, 12))
for i, column in enumerate(continuous_columns, 1):
    plt.subplot(3, 3, i)
    df[column].hist(bins=30, alpha=0.7, color='skyblue')
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=300)
plt.show()

# Boxplot: Glucose vs Diabetes outcome
plt.figure(figsize=(10, 6))
sns.boxplot(x='Outcome', y='Glucose', data=df, palette=['lightblue', 'lightcoral'])
plt.title('Glucose Levels by Diabetes Status')
plt.xticks([0, 1], ['Non-Diabetic', 'Diabetic'])
plt.tight_layout()
plt.savefig('glucose_comparison.png', dpi=300)
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.show()

# Analyze feature differences
diabetic_group = df[df['Outcome'] == 1]
non_diabetic_group = df[df['Outcome'] == 0]

print("\nFeatures with biggest differences between groups:")
feature_differences = {}
for feature in continuous_columns:
    difference = abs(diabetic_group[feature].mean() - non_diabetic_group[feature].mean())
    feature_differences[feature] = difference

sorted_differences = sorted(feature_differences.items(), key=lambda x: x[1], reverse=True)
for i, (feature, diff) in enumerate(sorted_differences[:3], 1):
    print(f"{i}. {feature}: difference = {diff:.3f}")

# =============================================================================
# PART C: FEATURE ENGINEERING
# =============================================================================

print("\nPART C: FEATURE ENGINEERING")
print("=" * 50)

# Create BMI categories
def categorize_bmi(bmi_value):
    if bmi_value < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi_value < 25:
        return 'Normal'
    elif 25 <= bmi_value < 30:
        return 'Overweight'
    else:
        return 'Obese'

df['BMI_Category'] = df_original['BMI'].apply(categorize_bmi)
print("BMI Category Distribution:")
print(df['BMI_Category'].value_counts())

# Interaction features
df['Glucose_BMI'] = df['Glucose'] * df['BMI']
df['Age_BMI'] = df['Age'] * df['BMI']
print("\nCreated interaction features: Glucose_BMI, Age_BMI")

# =============================================================================
# PART D: SECOND EDA & STATISTICAL INFERENCE
# =============================================================================

print("\nPART D: STATISTICAL INFERENCE")
print("=" * 50)

# T-test: Glucose levels by diabetes status
diabetic_glucose = df_original[df_original['Outcome'] == 1]['Glucose']
non_diabetic_glucose = df_original[df_original['Outcome'] == 0]['Glucose']
t_stat, p_value = stats.ttest_ind(diabetic_glucose, non_diabetic_glucose)

print(f"T-test Results: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
print("Significant difference in glucose levels" if p_value < 0.05 else "No significant difference")

# 95% Confidence interval for diabetes prevalence
n_patients = len(df)
n_diabetic = df['Outcome'].sum()
prevalence = n_diabetic / n_patients
z_score = st.norm.ppf(0.975)
std_error = np.sqrt((prevalence * (1 - prevalence)) / n_patients)
ci_lower = prevalence - z_score * std_error
ci_upper = prevalence + z_score * std_error

print(f"\nDiabetes Prevalence: {prevalence:.3f} ({prevalence*100:.1f}%)")
print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

# Statistical visualization
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.kdeplot(non_diabetic_glucose, label='Non-Diabetic', fill=True, alpha=0.6)
sns.kdeplot(diabetic_glucose, label='Diabetic', fill=True, alpha=0.6)
plt.title(f'Glucose Distribution (p-value: {p_value:.4f})')
plt.legend()

plt.subplot(1, 2, 2)
sns.boxplot(x='Outcome', y='Glucose', data=df_original)
plt.title('Glucose Levels by Diabetes Status')
plt.tight_layout()
plt.savefig('statistical_analysis.png', dpi=300)
plt.show()

# =============================================================================
# PART E: MACHINE LEARNING MODELING
# =============================================================================

print("\nPART E: MACHINE LEARNING MODELING")
print("=" * 50)

# Prepare data for ML
X = df.drop(['Outcome', 'BMI_Category'], axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Naive Bayes': GaussianNB()
}

results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Cross-validation for standard deviation
    cv_accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy').std()
    cv_precision = cross_val_score(model, X, y, cv=5, scoring='precision').std()
    cv_recall = cross_val_score(model, X, y, cv=5, scoring='recall').std()
    cv_f1 = cross_val_score(model, X, y, cv=5, scoring='f1').std()
    cv_auc = cross_val_score(model, X, y, cv=5, scoring='roc_auc').std()
    
    results[model_name] = {
        'accuracy': (accuracy, cv_accuracy),
        'precision': (precision, cv_precision),
        'recall': (recall, cv_recall),
        'f1': (f1, cv_f1),
        'roc_auc': (roc_auc, cv_auc),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

# Display results
print("\nMODEL PERFORMANCE RESULTS:")
print("=" * 80)
print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
print("-" * 80)

for model_name, metrics in results.items():
    acc_mean, acc_std = metrics['accuracy']
    prec_mean, prec_std = metrics['precision']
    rec_mean, rec_std = metrics['recall']
    f1_mean, f1_std = metrics['f1']
    auc_mean, auc_std = metrics['roc_auc']
    
    print(f"{model_name:<20} {acc_mean:.3f}±{acc_std:.3f}  {prec_mean:.3f}±{prec_std:.3f}  "
          f"{rec_mean:.3f}±{rec_std:.3f}  {f1_mean:.3f}±{f1_std:.3f}  {auc_mean:.3f}±{auc_std:.3f}")

# Confusion matrices and ROC curves
plt.figure(figsize=(15, 5))

# Confusion matrices
plt.subplot(1, 3, 1)
cm_lr = confusion_matrix(y_test, results['Logistic Regression']['y_pred'])
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression\nConfusion Matrix')

# ROC curves
plt.subplot(1, 3, 2)
for model_name, metrics in results.items():
    fpr, tpr, _ = roc_curve(y_test, metrics['y_pred_proba'])
    auc_score = metrics['roc_auc'][0]
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()

# Metric comparison
plt.subplot(1, 3, 3)
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
model_names = list(results.keys())
bar_width = 0.2
x_pos = np.arange(len(model_names))

colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
for i, metric in enumerate(metrics_to_plot):
    values = [results[model][metric][0] for model in model_names]
    plt.bar(x_pos + i * bar_width, values, bar_width, label=metric.capitalize(), color=colors[i])

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x_pos + bar_width * 1.5, model_names)
plt.legend()
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300)
plt.show()

print("\nANALYSIS COMPLETED SUCCESSFULLY!")