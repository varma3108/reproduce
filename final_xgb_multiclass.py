from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
from itertools import chain, cycle
import ast
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import seaborn as sns
import joblib


TARGET_CLASS = "CHM2210"


def get_data():
    df = pd.read_csv("full_set.csv")

    # converts strings to arrays
    df['Classes'] = df['Classes'].apply(ast.literal_eval)
    df['Semester Grades'] = df['Semester Grades'].apply(ast.literal_eval)
    df['Semester Points'] = df['Semester Points'].apply(ast.literal_eval)
    df['CRN'] = df['CRN'].apply(ast.literal_eval)

    return df


def preprocess_and_split_data(df, target_class, min_class_count):
    # Filter for students who took the target class
    pidms_with_target_class = df[df['Classes'].apply(lambda x: target_class in x)]['Pidm'].unique()
    df = df[df['Pidm'].isin(pidms_with_target_class)]
    df = df[['Pidm', 'Semester', 'HS GPA', 'Converted_SAT', 'Semester Points', 'Semester Grades', 'CRN', 'Classes']]

    # Find the first semester when the target class was taken
    def find_first_semester(student_df):
        target_row = student_df[student_df['Classes'].apply(lambda x: target_class in x)]
        if not target_row.empty:
            return target_row['Semester'].min()
        return None

    first_semester = df.groupby('Pidm').apply(find_first_semester).rename('Target_Semester')
    df = df.merge(first_semester, on='Pidm')

    # Filter all semesters before the target class was taken
    filtered_df = df[df['Semester'] <= df['Target_Semester']]

    # Extract grades and points for the target class
    def find_class_grades(student_df):
        for _, row in student_df.iterrows():
            if target_class in row['Classes']:
                index = row['Classes'].index(target_class)
                return row['Semester Points'][index], row['Semester Grades'][index]
        return None, None

    class_grades = filtered_df.groupby('Pidm').apply(find_class_grades).apply(pd.Series)
    class_grades.columns = ['Target_Points', 'Target_Grade']
    filtered_df = filtered_df.merge(class_grades, on='Pidm')

    # Remove rows with invalid grades
    filtered_df = filtered_df[~filtered_df['Target_Grade'].isin(['WE', 'IF', 'W', 'WC'])]
    filtered_df = filtered_df[filtered_df['Semester'] < filtered_df['Target_Semester']]

    # Aggregate data by student
    groupped_df = filtered_df.groupby('Pidm').agg({
        "HS GPA": 'first',
        'Converted_SAT': 'first',
        'Semester Grades': lambda x: sum(x, []),
        'Semester Points': lambda x: sum(x, []),
        'Classes': lambda x: sum(x, []),
        'CRN': lambda x: sum(x, []),
        'Target_Grade': 'first',
        'Target_Points': 'first',
    }).reset_index()

    # Create one-hot encoding for all classes
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

    # Split into train, dev, and test sets
    train, testing_data = train_test_split(one_hot_df, test_size=0.2, random_state=50)
    dev, test = train_test_split(testing_data, test_size=0.5, random_state=50)

    train_set = one_hot_df[one_hot_df.index.isin(train.index)]
    dev_set = one_hot_df[one_hot_df.index.isin(dev.index)]
    test_set = one_hot_df[one_hot_df.index.isin(test.index)]

    # Remove features with fewer than min_class_count observations
    columns_to_remove = []
    for column in train_set.columns:
        value_counts = train_set[column].value_counts()
        max_count = value_counts.max()
        non_max_count = value_counts.sum() - max_count
        if non_max_count <= min_class_count:
            columns_to_remove.append(column)

    train_set = train_set.drop(columns=columns_to_remove)
    dev_set = dev_set.drop(columns=columns_to_remove)
    test_set = test_set.drop(columns=columns_to_remove)

    # Integrate additional features
    train_set = train_set.join(groupped_df.set_index('Pidm')[['HS GPA', 'Converted_SAT', 'Target_Grade']])
    dev_set = dev_set.join(groupped_df.set_index('Pidm')[['HS GPA', 'Converted_SAT', 'Target_Grade']])
    test_set = test_set.join(groupped_df.set_index('Pidm')[['HS GPA', 'Converted_SAT', 'Target_Grade']])

    # Map grades to class labels
    grade_mapping = {
    'A+': 0, 'A': 0, 'A-': 0, 'S': 0,  # Class 0: A
    'B+': 1, 'B': 1, 'B-': 1,  # Class 1: B
    'C+': 2, 'C': 2, 'C-': 2,  # Class 2: C
    'D+': 3, 'D': 3, 'D-': 3, 'F': 3, 'U': 3  # Class 3: Fail
    }

    train_set['Target_Class'] = train_set['Target_Grade'].map(grade_mapping)
    dev_set['Target_Class'] = dev_set['Target_Grade'].map(grade_mapping)
    test_set['Target_Class'] = test_set['Target_Grade'].map(grade_mapping)

    # Drop rows with missing target classes
    train_set.dropna(subset=['Target_Class'], inplace=True)
    dev_set.dropna(subset=['Target_Class'], inplace=True)
    test_set.dropna(subset=['Target_Class'], inplace=True)

    # Separate features and targets
    X_train = train_set.drop(columns=['Target_Grade', 'Target_Class'])
    X_dev = dev_set.drop(columns=['Target_Grade', 'Target_Class'])
    X_test = test_set.drop(columns=['Target_Grade', 'Target_Class'])
    y_train = train_set['Target_Class'].astype(int)
    y_dev = dev_set['Target_Class'].astype(int)
    y_test = test_set['Target_Class'].astype(int)

    # Return processed data
    return X_train, y_train, X_dev, y_dev, X_test, y_test

def run_xgboost_model(X_train, y_train, X_dev, y_dev, X_test, y_test, params, class_names):
    """
    Train an XGBoost model, evaluate it, and visualize the confusion matrix.
    
    Parameters:
    X_train, y_train: Training features and labels.
    X_dev, y_dev: Validation features and labels.
    X_test, y_test: Test features and labels.
    params: Dictionary of XGBoost hyperparameters.
    class_names: List of class names for readability in the report.

    Returns:
    accuracy, class_report, auroc_score: Metrics.
    """
    # Initialize the XGBoost classifier with your parameters
    xgb_classifier = xgb.XGBClassifier(**params)

    # Fit the model to the training data
    eval_set = [(X_dev, y_dev)]
    xgb_classifier.fit(
        X_train, 
        y_train, 
        eval_set=eval_set
    )

    # --- STANDARD PREDICTIONS ---
    # Predict on the test set
    y_test_pred = xgb_classifier.predict(X_test)

    # Evaluate the model performance
    accuracy = accuracy_score(y_test, y_test_pred)
    conf_matrix = confusion_matrix(y_test, y_test_pred)

    # Modify classification report with class names
    class_report = classification_report(y_test, y_test_pred, target_names=class_names)

    # Print evaluation results
    print("--- Standard Evaluation ---")
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    # Visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # --- PROBABILITY-BASED METRICS (AUROC, P-R CURVE) ---
    
    # Get predicted probabilities
    y_test_proba = xgb_classifier.predict_proba(X_test)
    
    # Binarize y_test for multi-class OvR metrics
    n_classes = len(class_names)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    # Calculate Macro-Averaged AUROC Score
    auroc_score = roc_auc_score(y_test_bin, y_test_proba, multi_class='ovr', average='macro')
    print("\n--- Advanced Metrics ---")
    print(f"Macro-Averaged AUROC: {auroc_score:.2f}")

    # --- Plot AUC-ROC Curves (One-vs-Rest) ---
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_test_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of {class_names[i]} (area = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('auc_roc_curves.png')
    plt.close()

    # --- Plot Precision-Recall Curves (One-vs-Rest) ---
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_test_proba[:, i])
        average_precision[i] = average_precision_score(y_test_bin[:, i], y_test_proba[:, i])

    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'P-R curve of {class_names[i]} (AP = {average_precision[i]:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multi-class Precision-Recall Curve')
    plt.legend(loc="best")
    plt.savefig('precision_recall_curves.png')
    plt.close()

    # --- FEATURE IMPORTANCE ---
    
    # Get feature importances from the trained model
    feature_importances = xgb_classifier.feature_importances_

    # Assuming your feature names are in X_train.columns
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Display the top 20 features
    print("\n--- Feature Importance ---")
    print("Top 20 Features by Importance:")
    print(importance_df.head(20))

    # Plot the top 20 feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
    plt.title('Top 20 Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout() # Added to prevent labels from being cut off
    plt.savefig('feature_importance.png')
    plt.close()

    return accuracy, class_report, auroc_score


def save_xgboost_model(X_train, y_train, X_dev, y_dev, params, target_class):
    print(f"Training and saving model for class {target_class}...")
    
    # Initialize the XGBoost classifier
    xgb_classifier = xgb.XGBClassifier(**params)

    # Fit the model
    eval_set = [(X_dev, y_dev)]
    xgb_classifier.fit(
        X_train, 
        y_train, 
        eval_set=eval_set, 
        verbose=False
    )

    # Save the model
    model_path = f"./MLP_model_for_{target_class}.h5"
    joblib.dump(xgb_classifier, model_path)
    print(f"Model for class {target_class} saved to {model_path}")


    

# Define XGBoost parameters
xgboost_params = {
    'objective': 'multi:softmax',  # Multi-class classification
    'num_class': 4,               # Number of classes
    'colsample_bytree': 0.8,
    'learning_rate': 0.01,
    'max_depth': 10,
    'n_estimators': 500,
    'subsample': 0.8,
    'seed': 42,
    'early_stopping_rounds': 50,
    'eval_metric': 'mlogloss'
}

# Define class names for better readability
class_labels = ['A class', 'B class', 'C class', 'Fail']

# --- Main Execution ---
print("Loading data...")
df = get_data()

print("Preprocessing and splitting data...")
# Call the preprocessing function to get datasets
X_train, y_train, X_dev, y_dev, X_test, y_test = preprocess_and_split_data(df, TARGET_CLASS, 20)

print("Running XGBoost model and generating evaluations...")
# Run the XGBoost model
accuracy, class_report, auroc_score = run_xgboost_model(
    X_train, y_train, X_dev, y_dev, X_test, y_test, xgboost_params, class_labels
)

print("\n--- SCRIPT FINISHED ---")