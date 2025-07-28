import yaml
import logging
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from functools import partial

from feature_extraction.data_io import DataIO
from feature_extraction.preprocessing import DataProcessor
from feature_extraction.extractors.corr_extractor import CorrFeatureExtractor
from feature_extraction.extractors.ta_extractor import TAFeatureExtractor
from feature_extraction.extractors.slt_extractor import SLTExtractor
from feature_extraction.extractors.thesis_extractor import ThesisExtractor


# Mapping of feature extractor names (as strings) to their classes
EXTRACTOR_MAPPING = {
    "CorrFeatureExtractor": CorrFeatureExtractor,
    "TAFeatureExtractor": TAFeatureExtractor,
    "SLTFeatureExtractor": SLTExtractor, 
    "ThesisFeatureExtractor": ThesisExtractor,
}

# Mapping of attack keys (from config) to DataProcessor method names
ATTACK_FUNC_MAPPING = {
    "decorr": "apply_decorrelation_attack",
    "br": "apply_bias_removal",
    "ipd": "apply_ipd_jitter", 
    "pr": "apply_packet_reshaping",
    "tp": "apply_targeted_padding"
}

def _load_df(file_path: Path):
    """Loads a dataframe from a CSV file."""
    if not file_path.is_file():
        logging.error(f"CSV file not found at path: {file_path}")
        return None
    return pd.read_csv(file_path)

def worker(args):
    """
    Worker function to process one set of dataframes for one experiment.
    This function is executed in a separate process.
    """
    (i, bg_df_path, relay_df_path, gateway_df_path, pkt_limit,
     background_distributions_path, folder_name, experiment_config) = args
    
    logging.info(f"Worker processing item: {i} from folder {folder_name} for experiment '{experiment_config['name']}'")

    # Load dataframes from paths
    bg_df = _load_df(bg_df_path)
    relay_df = _load_df(relay_df_path)
    gateway_df = _load_df(gateway_df_path)

    if bg_df is None or relay_df is None or gateway_df is None:
        logging.warning(f"Skipping processing for item {i} due to missing data.")
        return None

    data_processor = DataProcessor(background_distributions_path)    
    
    # Dynamically create the list of attack functions using ATTACK_FUNC_MAPPING
    change_func_list = []
    for attack_config in experiment_config.get('attacks', []):
        attack_key = attack_config['name']
        
        # For baseline case
        if attack_key == "baseline":
            change_func_list.append(getattr(data_processor, "apply_bias_removal"))
            change_func_list.append(getattr(data_processor, "apply_decorrelation_attack"))
            continue
        
        # For other attacks
        params = attack_config.get('params', {})
        method_name = ATTACK_FUNC_MAPPING.get(attack_key)
        if method_name and hasattr(data_processor, method_name):
            func = getattr(data_processor, method_name)
            if params:
                change_func_list.append(partial(func, **params))
            else:
                change_func_list.append(func)
        else:
            logging.warning(f"Attack function for key '{attack_key}' not found in ATTACK_FUNC_MAPPING or DataProcessor. Skipping.")

    # Apply changes
    # bg_df_changed = data_processor.apply_changes(bg_df, pkt_limit, change_func_list)
    bg_df_changed = bg_df
    relay_df_changed = data_processor.apply_changes(relay_df, pkt_limit, change_func_list)
    
    if "decorr" in experiment_config.get('attacks', []):
        gateway_df_changed = data_processor.apply_changes(gateway_df, pkt_limit, change_func_list[:1])
    else:
        gateway_df_changed = data_processor.apply_changes(gateway_df, pkt_limit, change_func_list)

    # Instantiate the feature extractor
    extractor_name = experiment_config['feature_extractor']
    extractor_class = EXTRACTOR_MAPPING.get(extractor_name)
    if not extractor_class:
        logging.error(f"Unknown feature extractor: {extractor_name}")
        return None
        
    extractor_params = experiment_config.get('extractor_params', {})

    # Extract features for background and relay traffic
    # Handle CorrFeatureExtractor's unique signature
    if extractor_name == "CorrFeatureExtractor":
        bg_extractor = extractor_class(bg_df_changed, gateway_df_changed)
        relay_extractor = extractor_class(relay_df_changed, gateway_df_changed)
    else:
        bg_extractor = extractor_class(bg_df_changed)
        relay_extractor = extractor_class(relay_df_changed)

    bg_features = bg_extractor.process_df(pkt_limit)
    relay_features = relay_extractor.process_df(pkt_limit)
    
    # Add folder_name column
    if bg_features is not None and not bg_features.empty:
        bg_features['folder_name'] = folder_name
    if relay_features is not None and not relay_features.empty:
        relay_features['folder_name'] = folder_name
            
    return {"bg_features": bg_features, "relay_features": relay_features}

def process_batch(batch_num, bg_batch_paths, relay_batch_paths, gateway_batch_paths, 
                  folder_names_batch, data_io: DataIO, pkt_limit, 
                  background_distributions_path, experiment_config):
    """
    Processes a batch of dataframes in parallel for a single experiment.
    """
    tasks = [(i, bg_paths, relay_paths, gateway_paths, pkt_limit, 
              background_distributions_path, folder_name, experiment_config)
             for i, (bg_paths, relay_paths, gateway_paths, folder_name) in 
             enumerate(zip(bg_batch_paths, relay_batch_paths, gateway_batch_paths, folder_names_batch))]

    bg_features_dfs, relay_features_dfs = [], []

    with ProcessPoolExecutor() as executor:
        results = executor.map(worker, tasks)
        
        for result in results:
            if result:
                bg_features_dfs.append(result["bg_features"])
                relay_features_dfs.append(result["relay_features"])

    logging.info(f"Finished processing batch {batch_num} for experiment '{experiment_config['name']}'. Saving results.")
    
    experiment_name = experiment_config['name']
    data_io.save_bg_batch(bg_features_dfs, experiment_name, batch_num)
    data_io.save_relay_batch(relay_features_dfs, experiment_name, batch_num)

def process_split(split_data_io, batch_size, pkt_limit, background_distributions_path, experiment_config):
    """Processes a specific data split (train, test, or val) for a single experiment."""
    batch_iterator = split_data_io.load_batch_paths(batch_size)
    
    for i, (bg_batch, relay_batch, gateway_batch, folder_names) in enumerate(batch_iterator):
        logging.info(f"Starting batch number: {i} for experiment '{experiment_config['name']}'")
        process_batch(
            i, bg_batch, relay_batch, gateway_batch, folder_names, 
            split_data_io, pkt_limit, background_distributions_path, experiment_config
        )

def main():
    config_path = Path("configs/extraction_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logging.info(f"Loaded configuration from {config_path}")
        
    pkt_limit = config.get("pkt_limit", 20)
    batch_size = config.get("batch_size", 10)
    folder_path = Path(config.get("pcap_folder_path", "")).expanduser()
    csv_path = Path(config.get("csv_path", "")).expanduser()
    background_distributions_path = Path(config.get("background_distributions_path", ""))
    output_path = Path(config.get("output_path", ""))
    
    for p in [folder_path, csv_path, background_distributions_path, output_path]:
        if not str(p) or str(p) == ".":
            raise ValueError(f"Missing path configuration for: {p}")

    
    experiments = config.get("experiments", [])
    if not experiments:
        logging.warning("No experiments found in the configuration file.")
        return
        
    for exp_config in experiments:
        logging.info(f"===== Starting Experiment: {exp_config['name']} =====")
        
        # Save based on experiment last attack name
        last_attack_name = "none" if exp_config.get("attacks", "") == "" else exp_config.get("attacks")[-1]['name']
        train_data_io = DataIO(folder_path, csv_path, "train", output_path / last_attack_name / "train")
        test_data_io = DataIO(folder_path, csv_path, "test", output_path / last_attack_name / "test") 
        val_data_io = DataIO(folder_path, csv_path, "val", output_path / last_attack_name / "val")
        
        # Process each split for the current experiment
        for split_name, data_io in [("train", train_data_io), ("test", test_data_io), ("val", val_data_io)]:
            logging.info(f"--- Processing {split_name} data for experiment '{exp_config['name']}' ---")
            process_split(data_io, batch_size, pkt_limit, background_distributions_path, exp_config)
            logging.info(f"--- Finished processing {split_name} data for experiment '{exp_config['name']}' ---")

        logging.info(f"===== Finished Experiment: {exp_config['name']} =====")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        mp.set_start_method('spawn', force=True)
        logging.info("Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        pass  # It's safe to ignore if the start method has already been set.
    main()