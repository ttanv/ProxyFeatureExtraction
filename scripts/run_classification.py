"""
Script used to run the classification pipeline
"""

import yaml
import argparse
import pprint
import pandas as pd

from classification.data import get_data
from classification.models import XGBoostClassifier, SimpleCNN, SimpleTransformer
from classification.train import train_model
from classification.evaluate import evaluate_model

def main(config_path: str):
    """
    Main function to run the classification pipeline.
    """
    # 1. Load configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return
    
    print("--- Configuration ---")
    pprint.pprint(config)
    print("---------------------\n")

    # 2. Load data
    X_train, X_test, X_val, y_train, y_test, y_val = get_data(config)
    for df in [X_train, X_test, X_val, y_train, y_test, y_val]:
        df.drop(columns=['conn','folder_name'], inplace=True)

    # 3. Initialize model
    model_type = config['model_type']
    model = None

    if model_type == 'xgboost':
        params = config['training']['xgboost_params']
        model = XGBoostClassifier(params=params)
    
    elif model_type == 'cnn':
        params = config['training']['cnn_params']
        model = SimpleCNN(in_channels=params['in_channels'], num_classes=params['num_classes'])
    
    elif model_type == 'transformer':
        params = config['training']['transformer_params']
        model = SimpleTransformer(
            input_dim=params['input_dim'],
            num_heads=params['num_heads'],
            num_encoder_layers=params['num_encoder_layers'],
            num_classes=params['num_classes'],
            sequence_length=params['sequence_length']
        )
    else:
        raise ValueError(f"Unsupported model type in config: {model_type}")

    print(f"\nModel initialized: {model.__class__.__name__}\n")

    # 4. Train the model
    trained_model = train_model(model, X_train, X_val, y_train, y_val, config)

    # 5. Evaluate the model
    evaluate_model(trained_model, X_test, y_test, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a classification experiment.")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Path to the classification configuration YAML file."
    )
    args = parser.parse_args()
    main(args.config) 