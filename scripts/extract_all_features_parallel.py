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

def _load_df(file_path: Path):
    """Loads a dataframe from a CSV file."""
    if not file_path.is_file():
        # Log or handle the error as appropriate for your application
        logging.error(f"CSV file not found at path: {file_path}")
        return None  # Or raise an exception
    return pd.read_csv(file_path)

def worker(args):
    """
    Worker function to process one set of dataframes.
    This function is executed in a separate process.
    """
    # tp parameters
    n_packets_to_pad = 15
    pad_size = 50
    # ip parameters
    n_packets_to_jitter = 15
    max_delay_s = 0.02
    
    i, bg_df_path, relay_df_path, gateway_df_path, pkt_limit, background_distributions_path, folder_name = args
    logging.info(f"Worker processing item: {i} from folder {folder_name}")

    # Load dataframes from paths
    bg_df = _load_df(bg_df_path)
    relay_df = _load_df(relay_df_path)
    gateway_df = _load_df(gateway_df_path)

    # If any dataframe failed to load, skip processing
    if bg_df is None or relay_df is None or gateway_df is None:
        logging.warning(f"Skipping processing for item {i} due to missing data.")
        return None

    # Initialize processor within the worker to avoid serialization issues
    data_processor = DataProcessor(background_distributions_path)    
    
    # Create partials for attack functions
    padding_attack = partial(data_processor.apply_targeted_padding, 
                             n_packets_to_pad=n_packets_to_pad, 
                             pad_size=pad_size)
    
    ipd_jitter_attack = partial(data_processor.apply_ipd_jitter, 
                                n_packets_to_jitter=n_packets_to_jitter, 
                                max_delay_s=max_delay_s)
    
    change_func_list = [data_processor.apply_bias_removal]

    # Apply necessary changes
    bg_df_changed = data_processor.apply_changes(bg_df, pkt_limit, change_func_list)
    relay_df_changed = data_processor.apply_changes(relay_df, pkt_limit, change_func_list)
    gateway_df_changed = data_processor.apply_changes(gateway_df, pkt_limit, change_func_list)

    # Extract Corr features
    bg_corr_df = CorrFeatureExtractor(bg_df_changed, gateway_df_changed).process_df(pkt_limit)
    relay_corr_df = CorrFeatureExtractor(relay_df_changed, gateway_df_changed).process_df(pkt_limit)

    # Extract TA features
    bg_ta_df = TAFeatureExtractor(bg_df_changed).process_df(pkt_limit)
    relay_ta_df = TAFeatureExtractor(relay_df_changed).process_df(pkt_limit)
    
    # Add folder_name column to all resulting dataframes
    for df in [bg_ta_df, relay_ta_df, bg_corr_df, relay_corr_df]:
        if df is not None and not df.empty:
            df['folder_name'] = folder_name
            
    # Return a dictionary of results
    return {
        "bg_ta": bg_ta_df,
        "relay_ta": relay_ta_df,
        "bg_corr": bg_corr_df,
        "relay_corr": relay_corr_df,
    }

def process_batch(batch_num, bg_batch_paths, relay_batch_paths, gateway_batch_paths, 
                  folder_names_batch, data_io: DataIO, pkt_limit, background_distributions_path):
    """
    Processes a batch of dataframes in parallel.
    """
    tasks = []
    for i in range(len(bg_batch_paths)):
        # Package all arguments for the worker function
        tasks.append((
            i, bg_batch_paths[i], relay_batch_paths[i], gateway_batch_paths[i], 
            pkt_limit, background_distributions_path, folder_names_batch[i]
        ))

    # Lists to store results from workers
    bg_ta_dfs, relay_ta_dfs, bg_corr_dfs, relay_corr_dfs = [], [], [], []

    # Use ProcessPoolExecutor to run workers in parallel
    with ProcessPoolExecutor() as executor:
        results = executor.map(worker, tasks)
        
        # Process results as they are returned by the workers
        for result in results:
            if result is None:
                continue
            bg_ta_dfs.append(result["bg_ta"])
            relay_ta_dfs.append(result["relay_ta"])
            bg_corr_dfs.append(result["bg_corr"])
            relay_corr_dfs.append(result["relay_corr"])

    logging.info(f"Finished processing batch {batch_num}. Saving results.")
    # Save the collected results
    data_io.save_bg_batch(bg_ta_dfs, "ta", batch_num)
    data_io.save_bg_batch(bg_corr_dfs, "corr", batch_num)
    data_io.save_relay_batch(relay_ta_dfs, "ta", batch_num)
    data_io.save_relay_batch(relay_corr_dfs, "corr", batch_num)


def process_split(split_data_io, batch_size, pkt_limit, background_distributions_path):
    """Processes a specific data split (train, test, or val)."""
    batch_iterator = split_data_io.load_batch_paths(batch_size)
    
    for i, (bg_batch, relay_batch, gateway_batch, folder_names) in enumerate(batch_iterator):
        logging.info(f"Starting batch number: {i}")
        process_batch(
            i, bg_batch, relay_batch, gateway_batch, folder_names, 
            split_data_io, pkt_limit, background_distributions_path
        )
        

def main():
    # Load config from YAML
    config_path = Path("configs/ta_exp.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logging.info(f"Loaded the configuration from {config_path}")
        
    pkt_limit = config.get("pkt_limit", 20)
    batch_size = config.get("batch_size", 10)
    folder_path = Path(config.get("pcap_folder_path", ""))
    csv_path = Path(config.get("csv_path", ""))
    background_distributions_path = Path(config.get("background_distributions_path", ""))
    output_path = Path(config.get("output_path", ""))
    
    # Ensure all paths retrieved from yaml
    for p in [folder_path, csv_path, background_distributions_path, output_path]:
        if str(p) == "." or str(p) == "":
            raise ValueError(f"Missing path configuration for: {p}")
    
    # Create DataIO objects for each split
    train_data = DataIO(folder_path, csv_path, "train", output_path / "train")
    test_data = DataIO(folder_path, csv_path, "test", output_path / "test") 
    val_data = DataIO(folder_path, csv_path, "val", output_path / "val") 
    
    # Process each split
    logging.info("Created DataIO objects. Starting processing.")
    process_split(train_data, batch_size, pkt_limit, background_distributions_path)
    logging.info("Finished processing train data")
    
    process_split(test_data, batch_size, pkt_limit, background_distributions_path)
    logging.info("Finished processing test data")
    
    process_split(val_data, batch_size, pkt_limit, background_distributions_path)
    logging.info("Finished processing val data")


if __name__ == "__main__":
    # The if __name__ == "__main__": guard is essential for multiprocessing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        mp.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        # This will be raised if the start method has already been set.
        # It's safe to ignore in that case.
        pass
    main()