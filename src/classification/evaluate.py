"""
Module used to evaluate models
"""

import numpy as np
import torch
from sklearn.metrics import  f1_score, precision_score, recall_score, confusion_matrix
import pandas as pd

def evaluate_model(model, X_test, y_test, config: dict):
    """
    Evaluates a trained model on the test set.

    Args:
        model: The trained model instance.
        X_test: Test features.
        y_test: Test labels.
        config (dict): The configuration dictionary.
    """
    model_type = config['model_type']
    print(f"--- Evaluating {model_type} model ---")
    
    y_pred = None

    if model_type == 'xgboost':
        # Handle both binary and multi-class XGBoost models
        y_pred_raw = model.predict(X_test)
        
        # Check if predictions are probabilities (2D array) or labels (1D array)
        if hasattr(y_pred_raw, 'ndim') and y_pred_raw.ndim == 2:
            # Multi-class probabilities - convert to class labels
            y_pred = np.argmax(y_pred_raw, axis=1)
        else:
            # Already class labels
            y_pred = y_pred_raw
    
    elif model_type in ['cnn', 'transformer']:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        all_preds = []
        with torch.no_grad():
            test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            outputs = model(test_tensor)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
        y_pred = np.array(all_preds)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # --- Calculate and Print Metrics ---   
    metrics = {}
    
    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Determine if this is binary or multiclass classification
    num_classes = len(np.unique(y_test))
    
    if num_classes == 2:
        # Binary classification metrics
        tn, fp, fn, tp = cm.ravel()
        metrics['FPR'] = fp / (fp + tn)
        metrics['precision'] = precision_score(y_test, y_pred, pos_label=1)
        metrics['recall'] = recall_score(y_test, y_pred, pos_label=1)
        metrics['f1'] = f1_score(y_test, y_pred, pos_label=1)
    else:
        # Multiclass classification metrics (macro-averaged)
        metrics['precision'] = precision_score(y_test, y_pred, average='macro')
        metrics['recall'] = recall_score(y_test, y_pred, average='macro')
        metrics['f1'] = f1_score(y_test, y_pred, average='macro')
        
        # Calculate per-class metrics
        precision_per_class = precision_score(y_test, y_pred, average=None)
        recall_per_class = recall_score(y_test, y_pred, average=None)
        f1_per_class = f1_score(y_test, y_pred, average=None)
        
        print(f"\nPer-class metrics:")
        for i in range(num_classes):
            print(f"  Class {i}: Precision={precision_per_class[i]:.4f}, Recall={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}")
        
    print("Evaluation Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    
    if model_type == 'xgboost':
        print("\n--- Most Important Features (XGBoost) ---")
        
        # Get feature importances from the trained model
        importances = model.feature_importances_
        
        # Check if X_test is a pandas DataFrame to get feature names
        if isinstance(X_test, pd.DataFrame):
            feature_names = X_test.columns
            # Create a DataFrame for better visualization
            feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            # Sort the DataFrame by importance in descending order
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
            
            print("Top 10 most important features:")
            # Use to_string() for clean console output without the index
            print(feature_importance_df.head(5).to_string(index=False))
        else:
            # Fallback if X_test is not a DataFrame (e.g., a NumPy array)
            print("Feature importances could not be mapped to names (X_test is not a DataFrame).")
            # Sort importance scores and get the indices
            indices = np.argsort(importances)[::-1]
            print("Top 5 feature indices and their importance scores:")
            # Print top 5 features, or fewer if the model has less than 10
            for i in range(min(5, len(indices))):
                print(f"  Feature index {indices[i]}: {importances[indices[i]]:.4f}")

    # You could also save these to a file
    # results_df = pd.DataFrame([metrics])
    # results_df.to_csv(f"{config['data']['output_dir']}/evaluation_results.csv", index=False)

    return metrics