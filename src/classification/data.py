# src/classification/data.py
import pandas as pd
from pathlib import Path

from feature_extraction.preprocessing import DataProcessor

def get_full_df(feature_type: str, path: Path):
    bg_dfs = []
    rl_dfs = []
    for file in path.iterdir():
        filename = file.name
        if feature_type in filename and "bg" in filename:
            bg_dfs.append(pd.read_csv(path / filename))
        elif feature_type in filename and "relay" in filename:
            rl_dfs.append(pd.read_csv(path / filename))
    
    bg_df = pd.concat(bg_dfs, ignore_index=True)
    rl_df = pd.concat(rl_dfs, ignore_index=True)
    
    bg_df['label'] = 0
    rl_df['label'] = 1
    
    return pd.concat([bg_df, rl_df])

def get_feature_splits(feature_type: str, feature_type_2: str, path: Path, use_br: bool):
    # In case we are only dealing with a single feature
    train_path = Path('data/br') if use_br else path
    if feature_type_2 == '':
        train_df = get_full_df(feature_type, train_path / 'train')
        test_df = get_full_df(feature_type, path / "test")
        val_df = get_full_df(feature_type, path / "val")
    
        return train_df, test_df, val_df
    
    train_feature_1_df = get_full_df(feature_type, train_path / 'train')
    test_feature_1_df = get_full_df(feature_type, path / "test")
    val_feature_1_df = get_full_df(feature_type, path / "val")
    
    train_feature_2_df = get_full_df(feature_type_2, train_path / 'train')
    test_feature_2_df = get_full_df(feature_type_2, path / "test")
    val_feature_2_df = get_full_df(feature_type_2, path / "val")
    
    train_df = pd.merge(train_feature_1_df, train_feature_2_df.drop(columns=['label']), on=['folder_name', 'conn'])
    test_df = pd.merge(test_feature_1_df, test_feature_2_df.drop(columns=['label']), on=['folder_name', 'conn'])
    val_df = pd.merge(val_feature_1_df, val_feature_2_df.drop(columns=['label']), on=['folder_name', 'conn'])

    return train_df, test_df, val_df
    
    

def get_data(config: dict):
    """
    Loads, preprocesses, and splits the data for classification.
    """
    print("Loading data...")
    # Load the data from csvs
    input_path = config['data']['input_path']
    use_br = config['data']['use_br']
    feature_2 = config['data'].get('feature_2', '')
    train_df, test_df, val_df = get_feature_splits(config['data']['feature'], feature_2, Path(input_path), use_br)    

    # 2. (Optional) Apply attacks using your existing preprocessing module
    # if config['preprocessing']['apply_attacks']:
    #     print("Applying attacks...")
    #     df = apply_decorrelation_attack(df, config['preprocessing']['attack_params'])

    # 3. Separate features (X) and target (y)
    X_train, X_test, X_val = train_df.drop('label', axis=1), test_df.drop('label', axis=1), val_df.drop('label', axis=1)
    y_train, y_test, y_val = train_df['label'], test_df['label'], val_df['label']

    # The type of data returned can depend on the model
    if config['model_type'] == 'xgboost':
        # XGBoost can handle DataFrames directly
        return X_train, X_test, X_val, y_train, y_test, y_val
    else:
        return

