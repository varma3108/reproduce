import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    roc_auc_score,
    RocCurveDisplay,
    f1_score,
    roc_curve,
    auc
)
from sklearn.ensemble import (
    StackingClassifier, 
    AdaBoostClassifier, 
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils import resample

# --- 1. Import User's Data Pipeline ---
try:
    # Import the necessary data loading functions from your file
    from formatAndSplit import get_data, prep_data
    # Import your weighted undersample function
    from formatAndSplit import undersample as weighted_undersample
except ImportError as e:
    print(f"Error: Could not import from 'formatAndSplit.py'. [Error: {e}]")
    print("Please make sure 'run_experiments.py' is in the same folder as 'formatAndSplit.py'.")
    exit()
except Exception as e:
    print(f"An unknown error occurred during import: {e}")
    exit()

# --- 2. Import SMOTE ---
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("="*50)
    print("WARNING: 'imbalanced-learn' library not found.")
    print("SMOTE (Technique 3) will be skipped.")
    print("To install it, run: pip install imbalanced-learn")
    print("="*50)
    SMOTE = None # Set to None so we can skip it later


# --- 3. Define Fallback Function (to replace XGBoost) ---

def weighted_undersample_internal(X_train, y_train, minority_weight=1):
    """
    This is the logic from your formatAndSplit.py, defined
    internally for use by the findBestWeight_fallback.
    """
    counts = y_train.value_counts()
    fail_count = counts.get(0, 0)
    pass_count = counts.get(1, 1)
    
    # Calculate sample count based on the minority weight
    sample_count = int(minority_weight * fail_count + (1 - minority_weight) * pass_count)
    sample_count = min(sample_count, pass_count) # Ensure we don't try to sample more than we have
    
    # Separate the classes
    pass_class = X_train[y_train == 1]
    fail_class = X_train[y_train == 0]
    
    # Resample classes
    pass_sample = resample(pass_class, replace=False, n_samples=sample_count, random_state=50)
    fail_sample = resample(fail_class, replace=False, n_samples=fail_count, random_state=50) # Keep all fail
    
    # Combine resampled data
    X_balanced = pd.concat([pass_sample, fail_sample])
    y_balanced = pd.concat([y_train[pass_sample.index], y_train[fail_sample.index]])
    
    return X_balanced, y_balanced

def findBestWeight_fallback(X_train, y_train, step):
    """
    Internal fallback for findBestWeight that uses DecisionTree
    instead of the missing XGBoost.
    """
    print("Using internal fallback for findBestWeight (with DecisionTree).")
    best_f1 = -1
    best_weight = None

    # Use a 70% split of the training data as a dev set to find the weight
    xt, xd, yt, yd = train_test_split(X_train, y_train, test_size=0.7, random_state=50)

    for weight in np.arange(step, (1 + step), step):
        # Apply weighted undersampling to the internal training split
        X_resampled, y_resampled = weighted_undersample_internal(xt, yt, weight)

        # Use DecisionTree instead of XGBoost
        model = DecisionTreeClassifier(random_state=50, max_depth=9, class_weight='balanced')
        model.fit(X_resampled, y_resampled)
        y_pred = model.predict(xd) # Predict on the internal dev set
        f1 = f1_score(yd, y_pred, pos_label=0) # F1 for Fail class
        
        if f1 > best_f1:
            best_f1 = f1
            best_weight = weight
    
    if best_weight is None: 
        best_weight = 0.8 # A reasonable default based on your report's findings
    print(f"Internal findBestWeight found best_weight: {best_weight:.2f}")
    return best_weight


# --- 4. Define The 3 Sampling Techniques ---

# Technique 1: Undersampling (50:50)
def undersampling_5050(X, y):
    print("Applying 50:50 Undersampling...")
    counts = y.value_counts()
    fail_count = counts.get(0, 0)
    pass_count = counts.get(1, 1)
    sample_count = min(fail_count, pass_count)
    
    pass_class = X[y == 1]
    fail_class = X[y == 0]
    
    pass_sample = resample(pass_class, replace=False, n_samples=sample_count, random_state=50)
    fail_sample = resample(fail_class, replace=False, n_samples=sample_count, random_state=50)
    
    X_balanced = pd.concat([pass_sample, fail_sample])
    y_balanced = pd.concat([y[pass_sample.index], y[fail_sample.index]])
    print(f"50:50 Undersampling complete. New size: {len(y_balanced)}")
    return X_balanced, y_balanced

# Technique 2: Weighted Undersampling
def apply_weighted_undersampling(X, y):
    print("Applying Weighted Undersampling...")
    # Use our fallback function to find the weight
    best_weight = findBestWeight_fallback(X, y, 0.1) 
    print(f"Best weight found: {best_weight:.4f}")
    # Use the imported function from your file to apply it
    X_balanced, y_balanced = weighted_undersample(X, y, best_weight)
    print(f"Weighted Undersampling complete. New size: {len(y_balanced)}")
    return X_balanced, y_balanced

# Technique 3: SMOTE
def apply_smote(X, y):
    if SMOTE:
        print("Applying SMOTE...")
        sm = SMOTE(random_state=50)
        X_balanced, y_balanced = sm.fit_resample(X, y)
        print(f"SMOTE complete. New size: {len(y_balanced)}")
        return X_balanced, y_balanced
    else:
        print("Skipping SMOTE (imbalanced-learn library not found).")
        return None, None # Return None so we can skip it in the main loop


# --- 5. Define Model Combinations & Meta-Model ---

# Combination 1: Diverse Mix (from your report, XGB replaced with SVC)
combo_1 = [
    ('ada', AdaBoostClassifier(random_state=50)),
    ('gbc', GradientBoostingClassifier(random_state=50)),
    # SVC needs probability=True to work with AUROC and Stacking's predict_proba
    ('svc', SVC(probability=True, random_state=50, class_weight='balanced')),
    ('dt', DecisionTreeClassifier(random_state=50, max_depth=10, class_weight='balanced'))
]

# Combination 2: Fewer, Faster Models
combo_2 = [
    ('dt', DecisionTreeClassifier(random_state=50, max_depth=10, class_weight='balanced')),
    ('lr_base', LogisticRegression(solver='liblinear', class_weight='balanced', random_state=50)),
    ('ada', AdaBoostClassifier(random_state=50, n_estimators=50)) # Faster AdaBoost
]

# Combination 3: Boosting-Heavy (GBC and AdaBoost)
combo_3 = [
    ('gbc', GradientBoostingClassifier(random_state=50, n_estimators=150, max_depth=7)),
    ('ada', AdaBoostClassifier(random_state=50, n_estimators=100))
]

model_combos = [
    (combo_1, "Combo_1_Diverse"),
    (combo_2, "Combo_2_Fast"),
    (combo_3, "Combo_3_Boosting")
]

# Define the Meta-Model (Level 1)
# Using class_weight='balanced' to give a final focus on the minority class
meta_learner = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=50)


# --- 6. Define The Evaluation Function ---

def train_and_evaluate(model, X_train_res, y_train_res, X_test, y_test, model_name):
    """
    Trains a model, prints all metrics, and saves an AUROC curve plot.
    """
    print(f"\n========================================================")
    print(f"RUNNING EXPERIMENT: {model_name}")
    print(f"Training data shape: {X_train_res.shape}")
    print(f"Training data distribution:\n{y_train_res.value_counts(normalize=True)}")
    print(f"========================================================")
    
    try:
        # Train the model
        model.fit(X_train_res, y_train_res)
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] # Prob for class 1
        
        # --- Calculate and Print Metrics ---
        f1_fail = f1_score(y_test, y_pred, pos_label=0) # F1 for Fail class (0)
        acc = accuracy_score(y_test, y_pred)
        auroc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n--- RESULTS for {model_name} ---")
        print(f"🎯 F1-Score (Fail Class 0): {f1_fail:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"AUROC: {auroc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Fail (0)", "Pass (1)"]))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # --- Generate and Save AUROC Curve Plot ---
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc_val = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_val:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'AUROC Curve - {model_name}')
        plt.legend(loc="lower right")
        
        # Save the plot
        plot_filename = f"AUROC_{model_name}.png"
        plt.savefig(plot_filename)
        plt.close() # Close the figure to save memory
        
        print(f"\n✅ Saved AUROC plot to: {plot_filename}")
        print(f"--------------------------------------------------------\n")

    except Exception as e:
        print(f"\n❌ ERROR during training/evaluation for {model_name}:")
        print(f"{e}\n")


# --- 7. Run Main Experiment Loop ---
def main():
    print("Loading and preparing data...")
    df = get_data()
    # This creates the standard imbalanced splits from your pipeline
    X_train, X_dev, X_test, y_train, y_dev, y_test = prep_data(df, 'CHM2210', 10)

    print(f"Original X_train shape: {X_train.shape}")
    print(f"Original y_train distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test distribution:\n{y_test.value_counts(normalize=True)}\n")

    # --- Create the 3 Resampled Datasets ---
    resampled_data = []

    # Dataset 1: 50/50 Undersampling
    X_train_5050, y_train_5050 = undersampling_5050(X_train, y_train)
    resampled_data.append((X_train_5050, y_train_5050, "50_50_Undersample"))

    # Dataset 2: Weighted Undersampling
    X_train_weighted, y_train_weighted = apply_weighted_undersampling(X_train, y_train)
    resampled_data.append((X_train_weighted, y_train_weighted, "Weighted_Undersample"))

    # Dataset 3: SMOTE
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    if X_train_smote is not None:
        resampled_data.append((X_train_smote, y_train_smote, "SMOTE"))
    
    print("\n--- Starting All Model Experiments ---")

    # --- Iterate Over All Combinations ---
    for X_res, y_res, data_name in resampled_data:
        for combo, combo_name in model_combos:
            
            # Create a new StackingClassifier instance for this experiment
            stacking_model = StackingClassifier(
                estimators=combo,
                final_estimator=meta_learner,
                cv=3, # Using 3-fold CV for speed, especially since SVC is slow
                passthrough=True, # Meta-model sees original features + base predictions
                n_jobs=-1 # Use all cores
            )
            
            # Define a unique name for this run
            run_name = f"Stacking__{data_name}__{combo_name}"
            
            # Train and evaluate
            train_and_evaluate(stacking_model, X_res, y_res, X_test, y_test, run_name)

    print("\n\nAll experiments complete. Check the console output and the saved .png files.")

if __name__ == "__main__":
    main()