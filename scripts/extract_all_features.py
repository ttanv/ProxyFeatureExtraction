import yaml
import logging
from pathlib import Path
from feature_extraction.data_io import DataIO
from feature_extraction.preprocessing import DataProcessor
from feature_extraction.extractors.corr_extractor import CorrFeatureExtractor
from feature_extraction.extractors.ta_extractor import TAFeatureExtractor

def process_batch(batch_num, bg_batch, relay_batch, gateway_batch, 
                  data_io: DataIO,
                  data_processor: DataProcessor, 
                  pkt_limit):
    change_func_list = [data_processor.apply_bias_removal, data_processor.apply_attack]
    
    # Extract from each df from the current batch
    bg_ta_dfs = []
    relay_ta_dfs = []
    bg_corr_dfs = []
    relay_corr_dfs = []
    for i in range(0, len(bg_batch)):
        # Load and apply necessary changes
        logging.info(f"Currently processing df: {i}/{len(bg_batch)}")
        bg_df = data_processor.apply_changes(bg_batch[i], pkt_limit, change_func_list)
        relay_df = data_processor.apply_changes(relay_batch[i], pkt_limit, change_func_list)
        gateway_df = data_processor.apply_changes(gateway_batch[i], pkt_limit, change_func_list)
        logging.info("Finished applying changes including bias removal and attack")
        
        # Extract corr features
        bg_corr_extractor = CorrFeatureExtractor(bg_df, gateway_df)
        relay_corr_extractor = CorrFeatureExtractor(relay_df, gateway_df)
        bg_corr_dfs.append(bg_corr_extractor.process_df(pkt_limit))
        relay_corr_dfs.append(relay_corr_extractor.process_df(pkt_limit))
        
        # Extract TA features
        bg_ta_extractor = TAFeatureExtractor(bg_df)
        relay_ta_extractor = TAFeatureExtractor(relay_df)
        bg_ta_dfs.append(bg_ta_extractor.process_df(pkt_limit))
        relay_ta_dfs.append(relay_ta_extractor.process_df(pkt_limit))
        
    # Save the batches
    data_io.save_bg_batch(bg_ta_dfs, "ta", batch_num)
    data_io.save_bg_batch(bg_corr_dfs, "corr", batch_num)
    data_io.save_relay_batch(relay_ta_dfs, "ta", batch_num)
    data_io.save_relay_batch(relay_corr_dfs, "corr", batch_num)


def process_split(split_data_io, data_processor, batch_size, pkt_limit):
    """Processes specific split"""
    batch_iterator = split_data_io.load_batches(batch_size)
    
    for i, (bg_batch, relay_batch, gateway_batch) in enumerate(batch_iterator):
        logging.info(f"Starting batch number: {i}")
        process_batch(i, bg_batch, relay_batch, gateway_batch, split_data_io, data_processor, pkt_limit)
        

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
    for i in [pkt_limit, batch_size, folder_path, csv_path, background_distributions_path, output_path]:
        if isinstance(i, Path) and str(i) == "":
            raise ValueError("Missing some args or issue with loading yaml config file")
        elif not isinstance(i, Path) and i == "":
            raise ValueError("Missing some args or issue with loading yaml config file")
    
    # Create datasets
    train_data = DataIO(folder_path, csv_path, "train", output_path / "train")
    test_data = DataIO(folder_path, csv_path, "test", output_path / "test") 
    val_data = DataIO(folder_path, csv_path, "val", output_path / "val") 
    
    # Create preprocessing object
    data_processor = DataProcessor(background_distributions_path)
    
    # Process each split
    logging.info("Created DataIO and DataProcessor objects")
    process_split(train_data, data_processor, batch_size, pkt_limit)
    logging.info("Finished processing train data")
    
    process_split(test_data, data_processor, batch_size, pkt_limit)
    logging.info("Finished processing test data")
    
    process_split(val_data, data_processor, batch_size, pkt_limit)
    logging.info("Finsihed processing val data")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()