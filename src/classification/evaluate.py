"""
Module used to evaluate models
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
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
        y_pred = model.predict(X_test)
    
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
    if 'accuracy' in config['evaluation']['metrics']:
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
    if 'f1_score_macro' in config['evaluation']['metrics']:
        metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro')
    if 'precision_macro' in config['evaluation']['metrics']:
        metrics['precision_macro'] = precision_score(y_test, y_pred, average='macro')
    if 'recall_macro' in config['evaluation']['metrics']:
        metrics['recall_macro'] = recall_score(y_test, y_pred, average='macro')
        
    print("Evaluation Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    print(classification_report(y_test, y_pred))
    # You could also save these to a file
    # results_df = pd.DataFrame([metrics])
    # results_df.to_csv(f"{config['data']['output_dir']}/evaluation_results.csv", index=False)

    return metrics 