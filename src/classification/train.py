"""
Module used to train models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import joblib

from classification.models.xgboost import XGBoostClassifier

def train_model(model, X_train, X_val, y_train, y_val, config: dict):
    """
    Trains a given model on the provided data.

    Args:
        model: The model instance to train.
        X_train: Training features.
        y_train: Training labels.
        config (dict): The configuration dictionary.

    Returns:
        The trained model.
    """
    model_type = config['model_type']
    print(f"--- Starting training for {model_type} model ---")

    if model_type == 'xgboost':
        # XGBoost has a simple fit method
        model.fit(
            X_train, y_train,
            verbose=True,  # Print evaluation results
            eval_set=[(X_val, y_val)]
        )
    
    elif model_type in ['cnn', 'transformer']:
        # PyTorch models require a training loop
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Create DataLoader
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

        for epoch in range(config['training']['epochs']):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{config['training']['epochs']}, Loss: {running_loss/len(train_loader):.4f}")
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    print("--- Training finished ---")
    
    # --- Save the trained model ---
    output_dir = config['data']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_type}_model.pt") # .pt for torch, .joblib for others
    
    if model_type == 'xgboost':
        model_path = os.path.join(output_dir, f"{model_type}_model.json")
        model.save(model_path)
    else:
        torch.save(model.state_dict(), model_path)
        
    print(f"Model saved to {model_path}")

    return model

