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

def get_three_class_df(feature_type: str, bg_path: Path, relay_path: Path, gateway_path: Path):
    """
    Load and combine background, relay, and gateway data with proper labels.
    Labels: 0=background, 1=relay, 2=gateway
    """
    bg_dfs = []
    relay_dfs = []
    gateway_dfs = []
    
    # Load background files
    for file in bg_path.iterdir():
        filename = file.name
        if feature_type in filename and "bg" in filename:
            bg_dfs.append(pd.read_csv(bg_path / filename))
    
    # Load relay files  
    for file in relay_path.iterdir():
        filename = file.name
        if feature_type in filename and "relay" in filename:
            relay_dfs.append(pd.read_csv(relay_path / filename))
    
    # Load gateway files (gateway files are always ta, so only check for "gateway" in filename)
    for file in gateway_path.iterdir():
        filename = file.name
        if "gateway" in filename:
            gateway_dfs.append(pd.read_csv(gateway_path / filename))
    
    # Combine dataframes
    bg_df = pd.concat(bg_dfs, ignore_index=True) if bg_dfs else pd.DataFrame()
    relay_df = pd.concat(relay_dfs, ignore_index=True) if relay_dfs else pd.DataFrame()
    gateway_df = pd.concat(gateway_dfs, ignore_index=True) if gateway_dfs else pd.DataFrame()
    
    # Add labels only to non-empty dataframes
    if not bg_df.empty:
        bg_df['label'] = 0  # background
    if not relay_df.empty:
        relay_df['label'] = 1  # relay  
    if not gateway_df.empty:
        gateway_df['label'] = 2  # gateway
    
    return pd.concat([bg_df, relay_df, gateway_df], ignore_index=True)

def get_binary_gateway_vs_rest_df(feature_type: str, bg_path: Path, relay_path: Path, gateway_path: Path):
    """
    Load and combine data for binary classification: gateway vs (background + relay).
    Labels: 0=background+relay, 1=gateway
    """
    bg_dfs = []
    relay_dfs = []
    gateway_dfs = []
    
    # Load background files
    for file in bg_path.iterdir():
        filename = file.name
        if feature_type in filename and "bg" in filename:
            bg_dfs.append(pd.read_csv(bg_path / filename))
    
    # Load relay files  
    for file in relay_path.iterdir():
        filename = file.name
        if feature_type in filename and "relay" in filename:
            relay_dfs.append(pd.read_csv(relay_path / filename))
    
    # Load gateway files (gateway files are always ta, so only check for "gateway" in filename)
    for file in gateway_path.iterdir():
        filename = file.name
        if "gateway" in filename:
            gateway_dfs.append(pd.read_csv(gateway_path / filename))
    
    # Combine dataframes
    bg_df = pd.concat(bg_dfs, ignore_index=True) if bg_dfs else pd.DataFrame()
    relay_df = pd.concat(relay_dfs, ignore_index=True) if relay_dfs else pd.DataFrame()
    gateway_df = pd.concat(gateway_dfs, ignore_index=True) if gateway_dfs else pd.DataFrame()
    
    # Add labels: background and relay both get label 0, gateway gets label 1
    if not bg_df.empty:
        bg_df['label'] = 0
    if not relay_df.empty:
        relay_df['label'] = 0  
    if not gateway_df.empty:
        gateway_df['label'] = 1
    
    return pd.concat([bg_df, relay_df, gateway_df], ignore_index=True)

def get_feature_splits(feature_type: str, feature_type_2: str, path: Path, use_br: bool):
    # In case we are only dealing with a single feature
    train_path = Path('final_data/br') if use_br else path
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

def get_three_class_feature_splits(feature_type: str, feature_type_2: str = ''):
    """
    Get data splits for three-class classification (background, relay, gateway).
    Background and relay from 'br' folder, gateway from 'none' folder.
    """
    br_train_path = Path('final_data/br/train')
    br_test_path = Path('final_data/decorr/test') 
    br_val_path = Path('final_data/decorr/val')
    
    none_train_path = Path('final_data/none/train')
    none_test_path = Path('final_data/none/test')
    none_val_path = Path('final_data/none/val')
    
    if feature_type_2 == '':
        train_df = get_three_class_df(feature_type, br_train_path, br_train_path, none_train_path)
        test_df = get_three_class_df(feature_type, br_test_path, br_test_path, none_test_path)
        val_df = get_three_class_df(feature_type, br_val_path, br_val_path, none_val_path)
        
        return train_df, test_df, val_df
    
    # For dual feature types, merge on conn and folder_name
    train_feature_1_df = get_three_class_df(feature_type, br_train_path, br_train_path, none_train_path)
    test_feature_1_df = get_three_class_df(feature_type, br_test_path, br_test_path, none_test_path)
    val_feature_1_df = get_three_class_df(feature_type, br_val_path, br_val_path, none_val_path)
    
    train_feature_2_df = get_three_class_df(feature_type_2, br_train_path, br_train_path, none_train_path)
    test_feature_2_df = get_three_class_df(feature_type_2, br_test_path, br_test_path, none_test_path)
    val_feature_2_df = get_three_class_df(feature_type_2, br_val_path, br_val_path, none_val_path)
    
    train_df = pd.merge(train_feature_1_df, train_feature_2_df.drop(columns=['label']), on=['folder_name', 'conn'])
    test_df = pd.merge(test_feature_1_df, test_feature_2_df.drop(columns=['label']), on=['folder_name', 'conn'])
    val_df = pd.merge(val_feature_1_df, val_feature_2_df.drop(columns=['label']), on=['folder_name', 'conn'])

    return train_df, test_df, val_df

def get_binary_gateway_vs_rest_feature_splits(feature_type: str, feature_type_2: str = ''):
    """
    Get data splits for binary classification (gateway vs background+relay).
    Background and relay from 'br' folder, gateway from 'none' folder.
    """
    br_train_path = Path('final_data/br/train')
    br_test_path = Path('final_data/br/test') 
    br_val_path = Path('final_data/br/val')
    
    none_train_path = Path('final_data/none/train')
    none_test_path = Path('final_data/none/test')
    none_val_path = Path('final_data/none/val')
    
    if feature_type_2 == '':
        train_df = get_binary_gateway_vs_rest_df(feature_type, br_train_path, br_train_path, none_train_path)
        test_df = get_binary_gateway_vs_rest_df(feature_type, br_test_path, br_test_path, none_test_path)
        val_df = get_binary_gateway_vs_rest_df(feature_type, br_val_path, br_val_path, none_val_path)
        
        return train_df, test_df, val_df
    
    # For dual feature types, merge on conn and folder_name
    train_feature_1_df = get_binary_gateway_vs_rest_df(feature_type, br_train_path, br_train_path, none_train_path)
    test_feature_1_df = get_binary_gateway_vs_rest_df(feature_type, br_test_path, br_test_path, none_test_path)
    val_feature_1_df = get_binary_gateway_vs_rest_df(feature_type, br_val_path, br_val_path, none_val_path)
    
    train_feature_2_df = get_binary_gateway_vs_rest_df(feature_type_2, br_train_path, br_train_path, none_train_path)
    test_feature_2_df = get_binary_gateway_vs_rest_df(feature_type_2, br_test_path, br_test_path, none_test_path)
    val_feature_2_df = get_binary_gateway_vs_rest_df(feature_type_2, br_val_path, br_val_path, none_val_path)
    
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

