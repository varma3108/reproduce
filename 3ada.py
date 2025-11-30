import pandas as pd
import numpy as np
import ast
from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import (
    StackingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Set pandas options
pd.set_option('display.max_columns', None)

# --- 1. Data Loading and Preprocessing (from adav2.ipynb) ---
print("Loading and preparing data (from adav2.ipynb)...")
try:
    df = pd.read_csv("full_set.csv")
except FileNotFoundError:
    print("Error: 'full_set.csv' not found.")
    print("Please make sure this script is in the same directory as 'full_set.csv'")
    exit()

TARGET_CLASS = "CHM2210"

# Converts strings to arrays
df['Classes'] = df['Classes'].apply(ast.literal_eval)
df['Semester Grades'] = df['Semester Grades'].apply(ast.literal_eval)
df['Semester Points'] = df['Semester Points'].apply(ast.literal_eval)
df['CRN'] = df['CRN'].apply(ast.literal_eval)

# Filter for students who took the target class
Pidms_with_TARGET_CLASS = df[df['Classes'].apply(lambda x: TARGET_CLASS in x)]['Pidm'].unique()
df = df[df['Pidm'].isin(Pidms_with_TARGET_CLASS)]
df = df[['Pidm', 'Semester', 'HS GPA', 'Converted_SAT', 'Semester Points', 'Semester Grades', 'CRN', 'Classes']]

# Find the first semester with TARGET_CLASS for each student
def find_first_semester(student_df):
    chm2210_row = student_df[student_df['Classes'].apply(lambda x: TARGET_CLASS in x)]
    if not chm2210_row.empty:
        return chm2210_row['Semester'].min()
    return None

first_semester = df.groupby('Pidm').apply(lambda x: find_first_semester(x)).rename('Target_Semester')
df = df.merge(first_semester, on='Pidm')

# Filter all semesters after student took TARGET_CLASS
filtered_df = df[df['Semester'] <= df['Target_Semester']]

# Find grades/points for TARGET_CLASS
def find_class_grades(student_df):
    for _, row in student_df.iterrows():
        if TARGET_CLASS in row['Classes']:
            index = row['Classes'].index(TARGET_CLASS)
            return row['Semester Points'][index], row['Semester Grades'][index]
    return None, None

class_grades = filtered_df.groupby('Pidm').apply(lambda x: find_class_grades(x)).apply(pd.Series)
class_grades.columns = ['Target_Points', 'Target_Grade']
final_df = filtered_df.merge(class_grades, on='Pidm')

# Filter out non-grade entries
final_df = final_df[~final_df['Target_Grade'].isin(['WE', 'IF', 'W', 'WC'])]

# Filter to semesters *before* the target class
final_df = final_df[final_df['Semester'] < final_df['Target_Semester']]

# Group by student
groupped_df = final_df.groupby('Pidm').agg({
    "HS GPA": 'first',
    'Converted_SAT': 'first',
    'Semester Grades': lambda x: sum(x, []),
    'Semester Points': lambda x: sum(x, []),
    'Classes': lambda x: sum(x, []),
    'CRN': lambda x: sum(x, []),
    'Target_Grade': 'first',
    'Target_Points': 'first',
}).reset_index()

# One-hot encoding for classes
all_classes = sorted(set(chain.from_iterable(groupped_df['Classes'])))
def create_one_hot(classes, points, all_classes):
    one_hot_vector = [-1] * len(all_classes)
    for class_name, point in zip(classes, points):
        if class_name in all_classes:
            one_hot_vector[all_classes.index(class_name)] = point
    return one_hot_vector

groupped_df['One_Hot_Classes'] = groupped_df.apply(
    lambda row: create_one_hot(row['Classes'], row['Semester Points'], all_classes), axis=1
)
one_hot_df = pd.DataFrame(groupped_df['One_Hot_Classes'].tolist(), columns=all_classes, index=groupped_df['Pidm'])

# --- 2. Data Splitting (from adav2.ipynb) ---
# Create train, dev, and test sets
train, testing_data = train_test_split(one_hot_df, test_size=0.2, random_state=50)
dev, test = train_test_split(testing_data, test_size=0.5, random_state=50)

train_set = one_hot_df[one_hot_df.index.isin(train.index)]
dev_set = one_hot_df[one_hot_df.index.isin(dev.index)]
test_set = one_hot_df[one_hot_df.index.isin(test.index)]

# Filter columns (from adav2.ipynb - min_class_count=0)
columns_to_remove = []
for column in train_set.columns:
    value_counts = train_set[column].value_counts()
    max_count = value_counts.max()
    non_max_count = value_counts.sum() - max_count
    if non_max_count <= 0: # Your adav2.ipynb logic
        columns_to_remove.append(column)

train_set = train_set.drop(columns=columns_to_remove)
dev_set = dev_set.drop(columns=columns_to_remove)
test_set = test_set.drop(columns=columns_to_remove)

# Create pass/fail column
def map_pass_fail(grade):
    fail_grades = ['F', 'IF', 'W', 'D-', 'F', 'D+', 'D#', 'D+', 'F#', 'D', 'D', 'D-', 'U', 'W', 'F*', 'D*', 'CF', 'I', 'FF', 'Z', 'W*', 'F+', 'F-', 'F#', 'F*', 'D-*', 'IF', 'IF*', 'D+*', 'CIF', 'Z*', 'IU', 'M', 'CI', 'MU', 'U*', 'ID', 'IB', 'IU*', 'IS', 'CW']
    return 0 if grade in fail_grades else 1  # 0 = fail, 1 = pass

# Add supplementary columns (HS GPA, SAT, Target_Grade)
def add_sup_cols(df_to_join, original_groupped_df):
    groupped_filtered = original_groupped_df[original_groupped_df['Pidm'].isin(df_to_join.index)].set_index('Pidm')
    add_cols = ['HS GPA', 'Converted_SAT', 'Target_Grade']
    df_joined = df_to_join.join(groupped_filtered[add_cols])
    df_joined['pass_fail'] = df_joined['Target_Grade'].apply(map_pass_fail)
    return df_joined

train_set = add_sup_cols(train_set, groupped_df)
dev_set = add_sup_cols(dev_set, groupped_df)
test_set = add_sup_cols(test_set, groupped_df)

# --- 3. Create Final X/y splits for Train, Dev, and Test ---
# This is crucial: we evaluate on the *original, imbalanced* test set
def create_X_y(df):
    X = df.drop(columns=['Target_Grade', 'pass_fail'])
    X = X.dropna()
    y = df.loc[X.index, 'pass_fail']
    return X, y

X_train, y_train = create_X_y(train_set)
X_dev, y_dev = create_X_y(dev_set)
X_test, y_test = create_X_y(test_set) # The final, unseen test set

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_dev shape: {X_dev.shape}, y_dev shape: {y_dev.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print("\nTraining Set Class Distribution (Original):")
print(y_train.value_counts())

# --- 4. Weighted Undersampling (from formatAndSplit.py) ---
# We use the more advanced weighted undersampling from your report [cite: 1014]
# instead of the simple 50:50 split in adav2.ipynb
def undersample(X_train, y_train, minority_weight=1):
    counts = y_train.value_counts()
    fail_count = counts.get(0, 0)
    pass_count = counts.get(1, 1)
    
    # Calculate sample count based on the minority weight
    # This logic is slightly different from your file but achieves the goal
    sample_count = int(minority_weight * fail_count + (1 - minority_weight) * pass_count)
    
    # Ensure sample_count isn't larger than the pass count
    sample_count = min(sample_count, pass_count) 
    
    # If fail_count is 0, we can't sample, return original
    if fail_count == 0:
        return X_train, y_train

    # Separate the classes
    pass_class = X_train[y_train == 1]
    fail_class = X_train[y_train == 0]
    
    # Resample classes
    pass_sample = resample(pass_class, replace=False, n_samples=sample_count, random_state=50)
    fail_sample = fail_class # We keep all 'fail' samples
    
    # Combine resampled data
    X_balanced = pd.concat([pass_sample, fail_sample])
    y_balanced = pd.concat([y_train[pass_sample.index], y_train[fail_sample.index]])
    
    return X_balanced, y_balanced

def findBestWeight(X_train, y_train, step):
    # This function uses the DEV set to find the best weight
    print("\nFinding best undersampling weight...")
    best_f1 = -1
    best_weight = 0
    
    for weight in np.arange(step, (1 + step), step):
        X_resampled, y_resampled = undersample(X_train, y_train, weight)
        
        # Use a simple, fast model for finding the weight
        model = DecisionTreeClassifier(random_state=50, max_depth=5, class_weight='balanced')
        model.fit(X_resampled, y_resampled)
        
        y_pred = model.predict(X_dev) # Validate on the dev set
        f1 = f1_score(y_dev, y_pred, pos_label=0) # F1 for 'Fail' class
        
        if f1 > best_f1:
            best_f1 = f1
            best_weight = weight
            
    print(f"Best weight found: {best_weight:.2f} (F1-Fail: {best_f1:.4f} on dev set)")
    return best_weight

# Find the best weight and apply it to the training data
best_weight = findBestWeight(X_train, y_train, 0.1)
X_resampled, y_resampled = undersample(X_train, y_train, best_weight)

print(f"\nOriginal train size: {len(y_train)}, Resampled train size: {len(y_resampled)}")
print(f"Resampled class distribution:\n{y_resampled.value_counts()}")


# --- 5. Define and Train Stacking Classifier ---
print("\nDefining and training Stacking Classifier...")

# Base models (Level 0)
base_learners = [
    ('adav2', AdaBoostClassifier(
        # Using the best params from your adav2.ipynb grid search
        learning_rate=0.01, 
        n_estimators=50, 
        random_state=50
    )),
    ('gbc', GradientBoostingClassifier(
        random_state=50, 
        n_estimators=100,
        max_depth=5
    )),
    ('xgb', XGBClassifier(
        objective='binary:logistic',
        seed=50,
        n_estimators=100,
        use_label_encoder=False,
        eval_metric='logloss'
    )),
    ('dt', DecisionTreeClassifier(
        random_state=50,
        max_depth=7,
        class_weight='balanced'
    ))
]

# Meta-model (Level 1)
# We use class_weight='balanced' to force it to pay attention
# to the 'Fail' class, even at the meta-learning stage.
meta_learner = LogisticRegression(
    solver='liblinear', 
    class_weight='balanced', 
    random_state=50
)

# Initialize the Stacking Classifier
stacking_classifier = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,               # 5-fold cross-validation for base model predictions
    passthrough=True,   # Meta-model sees original features + base predictions
    n_jobs=-1           # Use all CPU cores
)

# Train the stack on the resampled data
stacking_classifier.fit(X_resampled, y_resampled)
print("Training complete.")

# --- 6. Evaluate on the UNSEEN TEST SET ---
print("\n--- 🚀 Stacking Model: TEST SET Evaluation ---")
# This evaluation is on X_test/y_test, which is imbalanced
# and has never been seen by the model.
y_pred_stack = stacking_classifier.predict(X_test)
y_pred_proba_stack = stacking_classifier.predict_proba(X_test)[:, 1]

print(f"Test Accuracy: {accuracy_score(y_test, y_pred_stack):.4f}")
print(f"Test AUROC: {roc_auc_score(y_test, y_pred_proba_stack):.4f}")

print("\nTest Classification Report (Stacking):")
# This is the most important output!
# Look at the F1-score and recall for 'Fail (0)'
print(classification_report(y_test, y_pred_stack, target_names=['Fail (0)', 'Pass (1)']))

print("\nTest Confusion Matrix (Stacking):")
print(confusion_matrix(y_test, y_pred_stack))


# --- 7. Comparative Model (Your AdaBoost Only) ---
print("\n--- (Comparison) AdaBoost Only: TEST SET Evaluation ---")
# We train your original model on the *same resampled data*
# and test it on the *same test set* for a fair comparison.
adav2_solo = AdaBoostClassifier(
    learning_rate=0.01, 
    n_estimators=50, 
    random_state=50
)
adav2_solo.fit(X_resampled, y_resampled)
y_pred_ada = adav2_solo.predict(X_test)

print("\nTest Classification Report (AdaBoost Only):")
print(classification_report(y_test, y_pred_ada, target_names=['Fail (0)', 'Pass (1)']))
print("\nTest Confusion Matrix (AdaBoost Only):")
print(confusion_matrix(y_test, y_pred_ada))