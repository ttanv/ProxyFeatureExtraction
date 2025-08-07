"""
Module for dealing with the PCAPs extracted data including both input and output operations.
This involves finding the folders, loading them into pandas dataframes, saving them. 
"""

import logging
from pathlib import Path
import pandas as pd

class DataIO:
    def __init__(self, folder_path, csv_path, split, output_path):
        self.folder_paths = self._get_folders(folder_path, csv_path, split)
        self.output_path = output_path
        
        return
    
    def _get_folders(self, folder_path, csv_path, split):
        """
        Function should use csv and split if provided, otherwise use all folders in path
        Should return a list of folders to be processed
        """
        # Get all subfolders in the input folder if csv_path not specified
        if not csv_path:
            return [p for p in folder_path.iterdir() if p.is_dir()]
        
        # Open csv to get folders to choose
        folders_df = pd.read_csv(csv_path)
        split_folders_df = folders_df[folders_df['split'] == split]
        split_folder_names = split_folders_df['folder_name'].tolist()
        
        split_folder_paths = [folder_path / folder for folder in split_folder_names]
        return split_folder_paths
    
    def _load_df(self, file_path):
        """Loads the gateway df file"""
        if not file_path.is_file():
            raise FileNotFoundError(f"CSV file not found at path: {file_path}")
        
        gateway_df = pd.read_csv(file_path)
        # if gateway_df.empty:
        #     raise ValueError(f"CSV file at {file_path} is empty.")
        
        return gateway_df
    
    def _load_batches(self, file_name, batch_size):
        """Returns an iterator of batches of dataframes"""
    
        for i in range(0, len(self.folder_paths), batch_size):
            batch_folders = self.folder_paths[i: i + batch_size]
            batch_dfs = [self._load_df(folder_name / file_name) for folder_name in batch_folders]
            yield batch_dfs
            
    def load_batches(self, batch_size):
        """Returns an iterator, each iteration has batch of bg, relay, and gateway dfs"""
        
        for i in range(0, len(self.folder_paths), batch_size):
            batch_folders = self.folder_paths[i: i + batch_size]
            gateway_batch_folders = [self._load_df(folder_name / "proxy_conn.csv") for folder_name in batch_folders]
            relay_batch_folders = [self._load_df(folder_name / "relayed_conn_labeled.csv") for folder_name in batch_folders]
            bg_batch_folders = [self._load_df(folder_name / "background_conn_labeled.csv") for folder_name in batch_folders]
            
            yield bg_batch_folders, relay_batch_folders, gateway_batch_folders
    
    def load_batch_paths(self, batch_size):
        """
        Returns an iterator, each iteration has a batch of bg, relay, gateway df paths, 
        and their corresponding folder names. More useful for parallelized workflows
        """
        
        for i in range(0, len(self.folder_paths), batch_size):
            batch_folders = self.folder_paths[i: i + batch_size]
            folder_names = [folder.name for folder in batch_folders]
            gateway_batch_paths = [folder_name / "proxy_conn.csv" for folder_name in batch_folders]
            relay_batch_paths = [folder_name / "relayed_conn_labeled.csv" for folder_name in batch_folders]
            bg_batch_paths = [folder_name / "background_conn_labeled.csv" for folder_name in batch_folders]
            
            yield bg_batch_paths, relay_batch_paths, gateway_batch_paths, folder_names
    
    def _save_batch(self, data_dfs, prefix, feature_type, batch_num):
        """Save a batch of data to CSV"""
        if not data_dfs:
            return
        
        batch_df = pd.concat(data_dfs)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = self.output_path / f'{prefix}_{feature_type}_batch_{batch_num}.csv'
        batch_df.to_csv(output_file, index=False)
        logging.info(f"Saved {prefix} batch {batch_num} with {len(batch_df)} rows")
        
    def save_bg_batch(self, data_dfs, feature_type, batch_num):
        """Save background batches of feature_type"""

        return self._save_batch(data_dfs, "bg", feature_type, batch_num)
    
    def save_relay_batch(self, data_dfs, feature_type, batch_num):
        """Save relay batches of feature_type"""
        
        return self._save_batch(data_dfs, "relay", feature_type, batch_num)
    
    def save_gateway_batch(self, data_dfs, feature_type, batch_num):
        """Save relay batches of feature_type"""
        
        return self._save_batch(data_dfs, "gateway", feature_type, batch_num)

    