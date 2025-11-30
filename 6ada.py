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
    # We alias it to be clear it's the one from your file
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

# --- 3. Import XGBoost (NEW) ---
try:
    from xgboost import XGBClassifier
    print("XGBoost library found.")
except ImportError:
    print("="*50)
    print("WARNING: 'xgboost' library not found.")
    print("XGBoost will not be used for weight finding or stacking.")
    print("To install it, run: pip install xgboost")
    print("="*50)
    XGBClassifier = None # Set to None so we can skip it

# --- 4. Define XGBoost-based Weight Finding Function (REPLACED) ---

def findBestWeight_xgb(X_train, y_train, step):
    """
    Finds the best undersampling weight using XGBoost on a dev split.
    """
    if XGBClassifier is None:
        print("WARNING: XGBoost not found. Cannot run findBestWeight_xgb.")
        print("Returning default weight 0.8.")
        return 0.8
        
    print("Using XGBoost to find best undersampling weight...")
    best_f1 = -1
    best_weight = None

    # Use a 70% split of the training data as a dev set to find the weight
    xt, xd, yt, yd = train_test_split(X_train, y_train, test_size=0.7, random_state=50)

    for weight in np.arange(step, (1 + step), step):
        # Apply weighted undersampling (from your file) to the internal training split
        X_resampled, y_resampled = weighted_undersample(xt, yt, weight)

        # Use XGBoost
        model = XGBClassifier(
            random_state=50, 
            eval_metric='logloss', 
            use_label_encoder=False # Suppress deprecation warning
        )
        model.fit(X_resampled, y_resampled)
        y_pred = model.predict(xd) # Predict on the internal dev set
        f1 = f1_score(yd, y_pred, pos_label=0) # F1 for Fail class
        
        if f1 > best_f1:
            best_f1 = f1
            best_weight = weight
    
    if best_weight is None: 
        best_weight = 0.8 # A reasonable default
    
    print(f"XGBoost findBestWeight found best_weight: {best_weight:.2f}")
    return best_weight


# --- 5. Define The 3 Sampling Techniques ---

# Technique 1: Undersampling (50:50)
def undersampling_5050(X, y):
    print("Applying 50:50 Undersampling...")
    counts = y.value_counts()
    fail_count = counts.get(0, 0)
    pass_count = counts.get(1, 1)
    
    # This is the crucial part: find the minimum count
    sample_count = min(fail_count, pass_count)
    
    if sample_count == 0:
        print("Warning: One class has 0 samples. Cannot balance.")
        return X, y

    pass_class = X[y == 1]
    fail_class = X[y == 0]
    
    pass_sample = resample(pass_class, replace=False, n_samples=sample_count, random_state=50)
    fail_sample = resample(fail_class, replace=False, n_samples=sample_count, random_state=50)
    
    X_balanced = pd.concat([pass_sample, fail_sample])
    # Ensure y_balanced aligns with the indices of X_balanced
    y_balanced = y.loc[X_balanced.index]
    
    print(f"50:50 Undersampling complete. New size: {len(y_balanced)} (Class 0: {sample_count}, Class 1: {sample_count})")
    return X_balanced, y_balanced

# Technique 2: Weighted Undersampling (UPDATED)
def apply_weighted_undersampling(X, y):
    print("Applying Weighted Undersampling...")
    # Use our new XGBoost-based function to find the weight
    best_weight = findBestWeight_xgb(X, y, 0.1) 
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


# --- 6. Define Model Combinations & Meta-Model (UPDATED) ---

# Define the base combinations
combo_1 = [
    ('ada', AdaBoostClassifier(random_state=50)),
    ('gbc', GradientBoostingClassifier(random_state=50)),
    ('svc', SVC(probability=True, random_state=50, class_weight='balanced')),
    ('dt', DecisionTreeClassifier(random_state=50, max_depth=10, class_weight='balanced'))
]

combo_2 = [
    ('dt', DecisionTreeClassifier(random_state=50, max_depth=10, class_weight='balanced')),
    ('lr_base', LogisticRegression(solver='liblinear', class_weight='balanced', random_state=50)),
    ('ada', AdaBoostClassifier(random_state=50, n_estimators=50))
]

combo_3 = [
    ('gbc', GradientBoostingClassifier(random_state=50, n_estimators=150, max_depth=7)),
    ('ada', AdaBoostClassifier(random_state=50, n_estimators=100))
]

# Conditionally add XGBoost if it was imported successfully
if XGBClassifier:
    print("Adding XGBClassifier to model combinations.")
    xgb_estimator = ('xgb', XGBClassifier(
        random_state=50, 
        eval_metric='logloss', 
        use_label_encoder=False
    ))
    combo_1.append(xgb_estimator)
    combo_2.append(xgb_estimator)
    combo_3.append(xgb_estimator)
else:
    print("Running combinations without XGBClassifier.")


model_combos = [
    (combo_1, "Combo_1_Diverse_XGB"),
    (combo_2, "Combo_2_Fast_XGB"),
    (combo_3, "Combo_3_Boosting_XGB")
]

# Define the Meta-Model (Level 1)
meta_learner = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=50)


# --- 7. Define The Evaluation Function ---

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
        
        # Handle probability prediction for AUROC
        # Check if model has predict_proba, else use decision_function or fail gracefully
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1] # Prob for class 1
            auroc = roc_auc_score(y_test, y_pred_proba)
        else:
            print("Warning: Model does not have 'predict_proba'. AUROC will be 0.0")
            y_pred_proba = y_pred # Fallback for plotting
            auroc = 0.0

        # --- Calculate and Print Metrics ---
        f1_fail = f1_score(y_test, y_pred, pos_label=0) # F1 for Fail class (0)
        acc = accuracy_score(y_test, y_pred)
        
        
        print(f"\n--- RESULTS for {model_name} ---")
        print(f"🎯 F1-Score (Fail Class 0): {f1_fail:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"AUROC: {auroc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Fail (0)", "Pass (1)"]))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # --- Generate and Save AUROC Curve Plot ---
        if auroc > 0.0: # Only plot if we have valid probabilities
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
        else:
            print("\nSkipped AUROC plot generation (no probabilities).")

        print(f"--------------------------------------------------------\n")

    except Exception as e:
        print(f"\n❌ ERROR during training/evaluation for {model_name}:")
        print(f"{e}\n")


# --- 8. Run Main Experiment Loop (MODIFIED) ---
def main():
    print("Loading and preparing data...")
    df = get_data()
    # This creates the standard imbalanced splits from your pipeline
    X_train, X_dev, X_test, y_train, y_dev, y_test = prep_data(df, 'CHM2210', 10)

    print(f"Original X_train shape: {X_train.shape}")
    print(f"Original y_train distribution:\n{y_train.value_counts(normalize=True)}")
    # Added prints for dev and test sets for clarity
    print(f"Original X_dev shape: {X_dev.shape}")
    print(f"Original y_dev distribution:\n{y_dev.value_counts(normalize=True)}\n") 
    print(f"Original X_test shape: {X_test.shape}")
    print(f"Original y_test distribution:\n{y_test.value_counts(normalize=True)}\n")

    # --- Create the 3 Resampled Training Datasets ---
    resampled_data = []

    # Dataset 1: 50/50 Undersampling (for TRAINING)
    X_train_5050, y_train_5050 = undersampling_5050(X_train, y_train)
    resampled_data.append((X_train_5050, y_train_5050, "Train_50_50_Undersample"))

    # Dataset 2: Weighted Undersampling (for TRAINING)
    X_train_weighted, y_train_weighted = apply_weighted_undersampling(X_train, y_train)
    resampled_data.append((X_train_weighted, y_train_weighted, "Train_Weighted_Undersample"))

    # Dataset 3: SMOTE (for TRAINING)
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    if X_train_smote is not None:
        resampled_data.append((X_train_smote, y_train_smote, "Train_SMOTE"))
    
    
    # --- (MODIFIED) Create a List of TEST Sets to Evaluate On ---
    test_sets = []
    
    # Test Set 1: The original, imbalanced test set
    test_sets.append((X_test, y_test, "Test_Imbalanced"))
    
    # *** NEWLY ADDED ***
    # Test Set 2: The development set 
    # This is the one that should have the 420/427 support from your image
    print("\n--- Adding Development Set to Test Loop ---")
    test_sets.append((X_dev, y_dev, "Test_Development_Set"))
    print("------------------------------------------\n")
    
    # Test Set 3: A new, 50:50 balanced test set (from original Test data)
    print("\n--- Creating 50:50 Balanced Test Set ---")
    X_test_5050, y_test_5050 = undersampling_5050(X_test, y_test)
    test_sets.append((X_test_5050, y_test_5050, "Test_Balanced_50_50"))
    print("------------------------------------------\n")

    
    print("\n--- Starting All Model Experiments ---")

    # --- (MODIFIED) Iterate Over All Test Sets ---
    for X_test_current, y_test_current, test_name in test_sets:
        
        print(f"\n========================================================")
        print(f"🧪 EVALUATING ON: {test_name} (Shape: {X_test_current.shape}) 🧪")
        # Added this print to show the support counts for the set being tested
        print(f"🧪 Support Counts:\n{y_test_current.value_counts()}")
        print(f"========================================================\n")

        # --- Iterate Over All Training Combinations ---
        for X_res, y_res, data_name in resampled_data:
            for combo, combo_name in model_combos:
                
                # Create a new StackingClassifier instance for this experiment
                stacking_model = StackingClassifier(
                    estimators=combo,
                    final_estimator=meta_learner,
                    cv=3, # Using 3-fold CV for speed
                    passthrough=True, # Meta-model sees original features + base predictions
                    n_jobs=-1 # Use all cores
                )
                
                # Define a unique name for this run to avoid overwriting files
                run_name = f"Stacking__{data_name}__{combo_name}__{test_name}"
                
                # Train and evaluate
                train_and_evaluate(
                    stacking_model, 
                    X_res,  # Training X
                    y_res,  # Training y
                    X_test_current, # Current Test X
                    y_test_current, # Current Test y
                    run_name
                )

    print("\n\nAll experiments complete. Check the console output and the saved .png files.")

if __name__ == "__main__":
    main()