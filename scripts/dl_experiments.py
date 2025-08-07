#!/usr/bin/env python3
"""
Traffic Correlation Experiments
Runs 5 experiments with different attack configurations on CorrTransformer and DeepCoFFEA models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
import math
import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import warnings
from datetime import datetime
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================
# HYPERPARAMETERS
# =====================================================
SEQ_LENGTH = 50          # Number of packets to look at in each flow
FEATURE_DIM = 3          # Input features for each packet: [Volume, Timestamp, Direction]
D_MODEL = 128            # Embedding dimension for the transformer
N_HEAD = 4               # Number of attention heads in the transformer
N_LAYERS = 4             # Number of layers in the transformer encoder
DROPOUT = 0.1            # Dropout rate
DROPOUT_PATH_RATE = 0.2
TOKEN_DROPOUT = 0.2
LEARNING_RATE = 1e-4     # Optimizer learning rate
BATCH_SIZE = 512         # Number of flow pairs per training batch
TRANSFORMER_EPOCHS = 50  # Number of training epochs for transformer
# DEEPCOFFEA_EPOCHS = 50   # Number of training epochs for DeepCoFFEA

# =====================================================
# MODEL COMPONENTS - TRANSFORMER
# =====================================================

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample"""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class StochasticDepthEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, dropout, drop_path_rate=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, src):
        x = src
        attn_output, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = src + self.drop_path1(self.dropout1(attn_output))
        
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(self.norm2(x)))))
        x = x + self.drop_path2(self.dropout2(ff_output))
        return x

class TokenDrop(nn.Module):
    """Randomly masks whole time‑steps; keeps sequence length."""
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        mask = (torch.rand(x.size(0), x.size(1), 1, device=x.device) > self.drop_prob).float()
        return x * mask / (1.0 - self.drop_prob)

class PositionalEncoding(nn.Module):
    """Adds positional information to the input embeddings."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class FlowCorrelationTransformer(nn.Module):
    """Cross-Attention Transformer model to correlate two traffic flows."""
    def __init__(self, feature_dim, d_model, n_head, n_layers, dropout, drop_path_rate=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.sep_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.token_drop = TokenDrop(drop_prob=TOKEN_DROPOUT)

        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=SEQ_LENGTH * 2 + 2)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        
        self.encoder_layers = nn.ModuleList([
            StochasticDepthEncoderLayer(
                d_model=d_model,
                n_head=n_head,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                drop_path_rate=dpr[i]
            )
            for i in range(n_layers)
        ])

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        self.attention_maps = []

    def _process_input(self, flow):
        """Processes raw flow data [Volume, Timestamp, Direction] to [Volume, DeltaTime, Direction]."""
        volumes = flow[..., 0] / 1500.0
        timestamps = flow[..., 1]
        direction = flow[..., 2]
        
        delta_times = torch.diff(timestamps, dim=1, prepend=timestamps[:, :1])
        
        delta_times = torch.clamp(delta_times, min=1e-6, max=10.0)
        mean_delta_times = delta_times.mean(dim=1, keepdim=True)
        delta_times = delta_times / (mean_delta_times + 1e-9)
        
        processed_flow = torch.stack([volumes, delta_times, direction], dim=-1)
        return self.input_projection(processed_flow)

    def forward(self, flow_a, flow_b):
        batch_size = flow_a.size(0)

        embedded_a = self.token_drop(self._process_input(flow_a))
        embedded_b = self.token_drop(self._process_input(flow_b))
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sep_tokens = self.sep_token.expand(batch_size, -1, -1)
        
        full_sequence = torch.cat([cls_tokens, embedded_a, sep_tokens, embedded_b], dim=1)
        
        full_sequence_pos = self.pos_encoder(full_sequence.permute(1, 0, 2)).permute(1, 0, 2)
        
        x = full_sequence_pos
        for layer in self.encoder_layers:
            x = layer(x)
        transformer_output = x
        
        cls_output = transformer_output[:, 0, :]
        
        score = self.output_head(cls_output)
        
        return score.squeeze(-1)


# =====================================================
# DATASET AND DATA HANDLING
# =====================================================

class FlowPairDataset(Dataset):
    """Custom PyTorch Dataset for flow pairs."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        flow_a, flow_b, label = self.data[idx]
        return (
            torch.tensor(flow_a, dtype=torch.float32),
            torch.tensor(flow_b, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )

# =====================================================
# DATA PROCESSOR AND ATTACKS
# =====================================================

class DataProcessor:
    """Encapsulates data modification logic and its required statistical parameters."""
    
    def __init__(self, json_path='packet_lengths.json'):
        """Initializes statistical values upon initialization"""
        (
            self.empirical_pkt_lens, self.bg_len_mean, self.bg_len_std,
            self.bg_tg_mean, self.bg_tg_std
        ) = self._load_empirical_samples(json_path)

    def _load_empirical_samples(self, json_path):
        """Load empirical packet length samples from JSON file"""
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
                empirical_pkt_lens = json_data['length']['empirical_samples']
                bg_len_mean = json_data['length']['mean']
                bg_len_std = json_data['length']['std']
                bg_tg_mean = json_data['timing']['mean']
                bg_tg_std = json_data['timing']['std']
                return empirical_pkt_lens, bg_len_mean, bg_len_std, bg_tg_mean, bg_tg_std
                
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            logging.warning(f"Warning: Could not load empirical samples: {e}")
            return None, None, None, None, None

    def apply_bias_removal(self, group, use_empirical_sampling=True):
        """Handles the bias removal steps"""
        if len(group) > 3 and group.iloc[3]['pkt_len'] > 1300:
            group.drop(index=group.index[3], inplace=True)
            if len(group) > 4:
                group.drop(index=group.index[4], inplace=True)

        if len(group) > 3:
            if use_empirical_sampling and (self.empirical_pkt_lens is not None):
                new_length = np.random.choice(self.empirical_pkt_lens)
            else:
                if self.bg_len_mean is None or self.bg_len_std is None:
                    raise ValueError("Background distribution parameters are required")
                mean_ = self.bg_len_mean
                std_ = self.bg_len_std
                new_length = np.random.normal(loc=mean_, scale=std_)
                new_length = max(1, int(round(new_length)))

            group.at[group.index[3], 'pkt_len'] = new_length

        return group

    def apply_decorrelation_attack(self, group):
        """Applies timing attack modification to the group"""
        if len(group) > 3:
            if self.bg_tg_mean is None or self.bg_tg_std is None:
                raise ValueError("Background timing distribution parameters are required")
            
            new_timing = np.random.lognormal(
                mean=self.bg_tg_mean,
                sigma=self.bg_tg_std
            )
            
            old_timing = group.iloc[3]['ts_relative'] - group.iloc[2]['ts_relative']
            timing_adjustment = old_timing - new_timing
            fourth_packet_idx = group.index[3]
            
            mask = group.index >= fourth_packet_idx
            group.loc[mask, 'ts_relative'] = group.loc[mask, 'ts_relative'] - timing_adjustment

        return group
    
    def apply_targeted_padding(self, group, n_packets_to_pad=10, pad_size=150):
        """Applies padding to the first n_packets_to_pad packets in the flow."""
        num_to_pad = min(len(group), n_packets_to_pad)
        
        if num_to_pad == 0:
            return group

        target_indices = group.index[:num_to_pad]
        
        padding_values = np.random.randint(1, pad_size + 1, size=num_to_pad)
        
        group.loc[target_indices, 'pkt_len'] += padding_values
        
        return group
    
    def apply_ipd_jitter(self, group, n_packets_to_jitter=15, max_delay_s=0.02):
        """Applies random timing jitter between the first n_packets_to_jitter packets."""
        num_to_jitter = min(len(group) -1, n_packets_to_jitter)
        
        if num_to_jitter <= 0:
            return group

        for i in range(num_to_jitter):
            jitter = np.random.uniform(0, max_delay_s)
            
            packet_idx = group.index[i]
            
            mask = group.index > packet_idx
            group.loc[mask, 'ts_relative'] += jitter
            
        return group
    
    def apply_packet_reshaping(self, group: pd.DataFrame,
                           split_threshold: int = 1000,
                           max_splits: int = 3,
                           min_pkt_size: int = 128) -> pd.DataFrame:
        """Structural obfuscation: split oversized packets into several random-length segments"""
        new_rows = []
        indices_to_drop = []

        for idx, row in group.iterrows():
            plen = row['pkt_len']
            if plen > split_threshold:
                indices_to_drop.append(idx)
                ts0 = row['ts_relative']

                k = np.random.randint(2, max_splits + 1)

                shares = np.random.dirichlet(np.ones(k))
                seg_lengths = np.maximum(
                    np.round(shares * (plen - k * min_pkt_size)).astype(int) + min_pkt_size,
                    min_pkt_size
                )
                seg_lengths[-1] = plen - seg_lengths[:-1].sum()

                for i, slen in enumerate(seg_lengths):
                    new_row = row.copy()
                    new_row['pkt_len'] = int(slen)
                    new_row['ts_relative'] = ts0 + i * 1e-6
                    new_rows.append(new_row)
            else:
                new_rows.append(row)

        reshaped = pd.DataFrame(new_rows).sort_values('ts_relative').reset_index(drop=True)
        return reshaped

    def apply_changes(self, df, pkt_limit, attack_func_list):
        """Main function that applies attacks from attack_func_list sequentially"""
        updated_groups = []

        if df.empty:
            return df

        for _, group in df.groupby('conn', sort=False):
            if len(group) >= pkt_limit:
                temp_group = group.sort_values(by='ts_relative', ascending=True).copy()
                
                for attack_func in attack_func_list:
                    temp_group = attack_func(temp_group)
                
                updated_groups.append(temp_group)
            else:
                updated_groups.append(group)

        if not updated_groups:
            raise ValueError(f"No valid groups found with minimum packet limit of {pkt_limit}")

        if len(updated_groups) == 0:
            logging.info("No changes applied, as no groups found")
            return df
        
        out_df = pd.concat(updated_groups).sort_index()
        return out_df

# =====================================================
# DATA LOADING FUNCTIONS
# =====================================================

def get_folders(folder_path, csv_path, split):
    """
    Function should use csv and split if provided, otherwise use all folders in path
    Should return a list of folders to be processed
    """
    # Get all subfolders in the input folder if csv_path not specified
    if not csv_path:
        folders = [os.path.join(folder_path, folder) 
                for folder in os.listdir(folder_path) 
                if os.path.isdir(os.path.join(folder_path, folder))]
        return folders
    
    # Open csv to get folders to choose
    folders_df = pd.read_csv(csv_path)
    split_folders_df = folders_df[folders_df['split'] == split]
    split_folder_names = split_folders_df['folder_name'].tolist()
    split_folder_paths = [os.path.join(folder_path, folder)
                     for folder in split_folder_names]
    
    return split_folder_paths

def get_pd_flow(data_processor, curr_df, gateway_df, label, pkt_limit, attack_funcs, training_data=False):
    """Get gateway conn for each curr_df that occured at same time"""
    conn_groups = curr_df.groupby('conn')
    trunc = 2000 if training_data else len(curr_df)
    
    # Get start and end time for each connection
    conn_time_ranges = conn_groups['ts_relative'].agg(['min', 'max']).reset_index()
    conn_time_ranges.columns = ['conn', 'start_time', 'end_time']
    
    results = []
    for _, row in conn_time_ranges.iterrows():
        
        conn = row['conn']
        start_time = row['start_time']
        end_time = row['end_time']
        
        # Find gateway rows within this time range
        mask = (gateway_df['ts_relative'] >= start_time) & (gateway_df['ts_relative'] <= end_time)
        
        # Get correct rows
        original_df = conn_groups.get_group(conn).copy()
        matching_df = gateway_df[mask].copy()

        # --- NEW: Determine and add direction feature ---
        # The first packet's source IP is considered the initiator (client)
        # Direction: 1.0 for client->server, -1.0 for server->client
        if not original_df.empty:
            client_ip_a = original_df.iloc[0]['src_ip']
            original_df['direction'] = np.where(original_df['src_ip'] == client_ip_a, 1.0, -1.0)
        else:
            original_df['direction'] = pd.Series(dtype=float)
            
        if not matching_df.empty:
            client_ip_a = matching_df.iloc[0]['src_ip']
            matching_df['direction'] = np.where(matching_df['src_ip'] == client_ip_a, 1.0, -1.0)
        else:
            matching_df['direction'] = pd.Series(dtype=float)


        # Apply the attacks on those rows only if its relayed traffic
        if label == 1 and not training_data:
            changes_func_list = [data_processor.apply_bias_removal]
            changes_func_list.extend(attack_funcs)
            original_df = data_processor.apply_changes(original_df, pkt_limit, changes_func_list)
            matching_df  = data_processor.apply_changes(matching_df, pkt_limit, changes_func_list[:1])
        elif label == 1:
            changes_func_list = [data_processor.apply_bias_removal]
            original_df = data_processor.apply_changes(original_df, pkt_limit, changes_func_list)
            matching_df  = data_processor.apply_changes(matching_df, pkt_limit, changes_func_list)
        
        # Get relevant rows as lists
        original_rows = np.array(list(zip(original_df['pkt_len'], original_df['ts_relative'], original_df['direction']))[0:pkt_limit])
        matching_rows = np.array(list(zip(matching_df['pkt_len'], matching_df['ts_relative'], matching_df['direction']))[0:pkt_limit])
        
        # Pad to ensure same len
        original_padded = np.pad(original_rows, 
                                 [[0, pkt_limit-len(original_rows)], [0, 0]], 
                                 mode='constant', 
                                 constant_values=0)
        
        # NEW: Handle empty matching_rows case before padding
        if len(matching_rows) == 0:
            matching_rows = np.array([[0, 0, 0]]) # Use a placeholder with 3 features

        matching_padded = np.pad(matching_rows, 
                                 [[0, pkt_limit-len(matching_rows)], [0, 0]], 
                                 mode='constant', 
                                 constant_values=0)
        
        results.append((original_padded, matching_padded, label))
        
    return results



def get_flows(folder_paths, data_processor, training_data=False):
    """Takes path to extracted folders, returns list of flow samples for path"""
    flows = []
    for folder_path in folder_paths:
        if "mixed" not in folder_path:
            continue
        
        # Get paths
        
        bg_file_path = os.path.join(folder_path, "background_conn_labeled.csv")
        rl_file_path = os.path.join(folder_path, "relayed_conn_labeled.csv")
        gateway_file_path = os.path.join(folder_path, "proxy_conn.csv")
        
        
        # Get pds
        try:
            bg_pd = pd.read_csv(bg_file_path)
            rl_pd = pd.read_csv(rl_file_path)
            gateway_pd = pd.read_csv(gateway_file_path)
        except:
            continue
        
        # Get flows
        bg_flows = get_pd_flow(data_processor, bg_pd, gateway_pd, 0, SEQ_LENGTH, training_data)
        rl_flows = get_pd_flow(data_processor, rl_pd, gateway_pd, 1, SEQ_LENGTH, training_data)
        
        print(f"Total bg flows extracted:  {len(bg_flows)}")
        print(f"Total relayed flows extracted: {len(rl_flows)}")
        
        # Add to flow list
        flows.extend(bg_flows)
        flows.extend(rl_flows)
        
    return flows
        

def process_single_folder(args):
    """Process a single folder and return its flows"""
    folder_path, data_processor_config, attack_funcs, training_data, seq_length = args
    
    # Create a new DataProcessor instance for this worker
    data_processor = DataProcessor(data_processor_config)
    
    if "mixed" not in folder_path:
        return []
    
    flows = []
    folder_name = os.path.basename(folder_path)
    
    # Get paths
    bg_file_path = os.path.join(folder_path, "background_conn_labeled.csv")
    rl_file_path = os.path.join(folder_path, "relayed_conn_labeled.csv")
    gateway_file_path = os.path.join(folder_path, "proxy_conn.csv")
    
    # Get pds
    try:
        bg_pd = pd.read_csv(bg_file_path)
        rl_pd = pd.read_csv(rl_file_path)
        gateway_pd = pd.read_csv(gateway_file_path)
    except Exception as e:
        print(f"Error processing folder {folder_name}: {e}")
        return []
    
    # Get flows
    try:
        bg_flows = get_pd_flow(data_processor, bg_pd, gateway_pd, 0, seq_length, attack_funcs, training_data)
        rl_flows = get_pd_flow(data_processor, rl_pd, gateway_pd, 1, seq_length, attack_funcs, training_data)
        
        print(f"Folder {folder_name}: bg={len(bg_flows)}, relayed={len(rl_flows)}")
        
        # Add to flow list
        flows.extend(bg_flows)
        flows.extend(rl_flows)
        
    except Exception as e:
        print(f"Error processing flows in folder {folder_name}: {e}")
        return []
    
    return flows

def get_flows_parallel(folder_paths, data_processor_config, attack_funcs, training_data=False, seq_length=None, n_processes=None):
    """Parallel version of get_flows using multiprocessing"""
    
    if n_processes is None:
        # Use most cores but leave a few free for the system
        n_processes = min(len(folder_paths), max(1, mp.cpu_count() - 2))
    
    print(f"Processing {len(folder_paths)} folders using {n_processes} processes...")
    
    # Filter folders that contain "mixed" upfront to avoid unnecessary work
    valid_folders = [folder for folder in folder_paths if "mixed" in folder]
    print(f"Found {len(valid_folders)} valid folders to process")
    
    if not valid_folders:
        return []
    
    # Prepare arguments for each worker
    args = [(folder_path, data_processor_config, attack_funcs, training_data, seq_length) 
            for folder_path in valid_folders]
    
    start_time = time.time()
    
    # Process folders in parallel
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_single_folder, args)
    
    # Flatten results
    flows = []
    for result in results:
        if result:  # Only extend if result is not empty
            flows.extend(result)
    
    end_time = time.time()
    print(f"Processed {len(valid_folders)} folders in {end_time - start_time:.2f} seconds")
    print(f"Total flows extracted: {len(flows)}")
    
    return flows

def process_connection_batch(args):
    """Process a batch of connections in parallel"""
    conn_batch, gateway_df, data_processor_config, label, pkt_limit, training_data = args
    
    # Create DataProcessor for this worker
    data_processor = DataProcessor(data_processor_config)
    
    results = []
    
    for _, row in conn_batch.iterrows():
        conn = row['conn']
        start_time = row['start_time']
        end_time = row['end_time']
        
        try:
            # Find gateway rows within this time range
            mask = (gateway_df['ts_relative'] >= start_time) & (gateway_df['ts_relative'] <= end_time)
            
            
            original_df = conn_batch.get_group(conn).copy()
            matching_df = gateway_df[mask].copy()

            # --- NEW: Determine and add direction feature ---
            # The first packet's source IP is considered the initiator (client)
            # Direction: 1.0 for client->server, -1.0 for server->client
            if not original_df.empty:
                client_ip_a = original_df.iloc[0]['src_ip']
                original_df['direction'] = np.where(original_df['src_ip'] == client_ip_a, 1.0, -1.0)
            else:
                original_df['direction'] = pd.Series(dtype=float)
                
            if not matching_df.empty:
                client_ip_a = matching_df.iloc[0]['src_ip']
                matching_df['direction'] = np.where(matching_df['src_ip'] == client_ip_a, 1.0, -1.0)
            else:
                matching_df['direction'] = pd.Series(dtype=float)


            # Apply the attacks on those rows only if its relayed traffic
            if label == 1 and not training_data:
                changes_func_list = [data_processor.apply_bias_removal, data_processor.apply_decorrelation_attack]
                original_df = data_processor.apply_changes(original_df, pkt_limit, changes_func_list)
                matching_df  = data_processor.apply_changes(matching_df, pkt_limit, changes_func_list[:1])
            elif label == 1:
                changes_func_list = [data_processor.apply_bias_removal]
                original_df = data_processor.apply_changes(original_df, pkt_limit, changes_func_list)
                matching_df  = data_processor.apply_changes(matching_df, pkt_limit, changes_func_list)
            
            # Get relevant rows as lists
            original_rows = np.array(list(zip(original_df['pkt_len'], original_df['ts_relative'], original_df['direction']))[0:pkt_limit])
            matching_rows = np.array(list(zip(matching_df['pkt_len'], matching_df['ts_relative'], matching_df['direction']))[0:pkt_limit])
            
            # Pad to ensure same len
            original_padded = np.pad(original_rows, 
                                    [[0, pkt_limit-len(original_rows)], [0, 0]], 
                                    mode='constant', 
                                    constant_values=0)
            
            # NEW: Handle empty matching_rows case before padding
            if len(matching_rows) == 0:
                matching_rows = np.array([[0, 0, 0]]) # Use a placeholder with 3 features

            matching_padded = np.pad(matching_rows, 
                                    [[0, pkt_limit-len(matching_rows)], [0, 0]], 
                                    mode='constant', 
                                    constant_values=0)
            
            results.append((original_padded, matching_padded, label))
            
        except Exception as e:
            print(f"Error processing connection {conn}: {e}")
            continue
    
    return results

def get_pd_flow_optimized(data_processor, curr_df, gateway_df, label, pkt_limit, training_data=False):
    """Optimized version of get_pd_flow with potential for connection-level parallelization"""
    
    if curr_df.empty:
        return []
    
    conn_groups = curr_df.groupby('conn')
    trunc = 2000 if training_data else len(curr_df)
    
    # Get start and end time for each connection
    conn_time_ranges = conn_groups['ts_relative'].agg(['min', 'max']).reset_index()
    conn_time_ranges.columns = ['conn', 'start_time', 'end_time']
    
    results = []
    
    # Pre-sort gateway_df for faster time-based filtering
    gateway_df_sorted = gateway_df.sort_values('ts_relative')
    
    for _, row in conn_time_ranges.iterrows():
        conn = row['conn']
        start_time = row['start_time']
        end_time = row['end_time']
        
        # Use binary search for faster time range filtering on large datasets
        start_idx = gateway_df_sorted['ts_relative'].searchsorted(start_time, side='left')
        end_idx = gateway_df_sorted['ts_relative'].searchsorted(end_time, side='right')
        
        # Get correct rows
        original_df = conn_groups.get_group(conn).copy()
        matching_df = gateway_df_sorted.iloc[start_idx:end_idx].copy()

        # Add direction feature
        if not original_df.empty:
            client_ip_a = original_df.iloc[0]['src_ip']
            original_df['direction'] = np.where(original_df['src_ip'] == client_ip_a, 1.0, -1.0)
        else:
            original_df['direction'] = pd.Series(dtype=float)
            
        if not matching_df.empty:
            client_ip_a = matching_df.iloc[0]['src_ip']
            matching_df['direction'] = np.where(matching_df['src_ip'] == client_ip_a, 1.0, -1.0)
        else:
            matching_df['direction'] = pd.Series(dtype=float)

        # Apply the attacks on those rows only if its relayed traffic
        if label == 1 and not training_data:
            changes_func_list = [data_processor.apply_bias_removal, data_processor.apply_decorrelation_attack]
            original_df = data_processor.apply_changes(original_df, pkt_limit, changes_func_list)
            matching_df = data_processor.apply_changes(matching_df, pkt_limit, changes_func_list[:1])
        elif label == 1:
            changes_func_list = [data_processor.apply_bias_removal]
            original_df = data_processor.apply_changes(original_df, pkt_limit, changes_func_list)
            matching_df = data_processor.apply_changes(matching_df, pkt_limit, changes_func_list)
        
        # Get relevant rows as lists
        original_rows = np.array(list(zip(original_df['pkt_len'], original_df['ts_relative'], original_df['direction']))[0:pkt_limit])
        matching_rows = np.array(list(zip(matching_df['pkt_len'], matching_df['ts_relative'], matching_df['direction']))[0:pkt_limit])
        
        # Pad to ensure same len
        original_padded = np.pad(original_rows, 
                                 [[0, pkt_limit-len(original_rows)], [0, 0]], 
                                 mode='constant', 
                                 constant_values=0)
        
        # Handle empty matching_rows case before padding
        if len(matching_rows) == 0:
            matching_rows = np.array([[0, 0, 0]])

        matching_padded = np.pad(matching_rows, 
                                 [[0, pkt_limit-len(matching_rows)], [0, 0]], 
                                 mode='constant', 
                                 constant_values=0)
        
        results.append((original_padded, matching_padded, label))
        
    return results

# Updated usage
def load_data_parallel(folder_path, csv_path, seq_length, attack_funcs, n_processes=None, ):
    """
    Main function to load all data in parallel
    """
    
    # Determine number of processes
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 2)
    
    print(f"Using {n_processes} processes for parallel data loading")
    
    # Load training data
    print("Loading training data...")
    train_folders = get_folders(folder_path, csv_path, "train")
    data_processor_config = "configs/background_distributions.json"
    train_flows = get_flows_parallel(train_folders, data_processor_config, attack_funcs, True, seq_length, n_processes)
    
    # Load validation data
    print("Loading validation data...")
    val_folders = get_folders(folder_path, csv_path, "val")
    val_flows = get_flows_parallel(val_folders, data_processor_config, attack_funcs, False, seq_length, n_processes)
    
    # Load test data
    print("Loading test data...")
    test_folders = get_folders(folder_path, csv_path, 'test')
    test_flows = get_flows_parallel(test_folders, data_processor_config, attack_funcs, False, seq_length, n_processes)
    
    return train_flows, val_flows, test_flows

# Memory-efficient batch processing for very large datasets
def load_data_in_batches(folder_path, csv_path, seq_length, batch_size=50, n_processes=None):
    """
    Load data in batches to manage memory usage for very large datasets
    """
    all_folders = {
        'train': get_folders(folder_path, csv_path, "train"),
        'val': get_folders(folder_path, csv_path, "val"),
        'test': get_folders(folder_path, csv_path, "test")
    }
    
    results = {'train': [], 'val': [], 'test': []}
    data_processor_config = "configs/background_distributions.json"
    
    for split_name, folders in all_folders.items():
        print(f"Processing {split_name} data in batches...")
        training_data = (split_name == 'train')
        
        # Process folders in batches
        for i in range(0, len(folders), batch_size):
            batch_folders = folders[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(folders)-1)//batch_size + 1}")
            
            batch_flows = get_flows_parallel(
                batch_folders, 
                data_processor_config, 
                training_data, 
                seq_length, 
                n_processes
            )
            results[split_name].extend(batch_flows)
            
            # Optional: save intermediate results to disk to free memory
            # pickle.dump(batch_flows, open(f'{split_name}_batch_{i//batch_size}.pkl', 'wb'))
    
    return results['train'], results['val'], results['test']

# Performance monitoring wrapper
def timed_load_data(folder_path, csv_path, seq_length, attack_funcs, n_processes=None):
    """
    Load data with detailed timing information
    """
    import psutil
    import time
    
    start_time = time.time()
    start_memory = psutil.virtual_memory().used / (1024**3)  # GB
    
    print(f"Starting data loading...")
    print(f"Initial memory usage: {start_memory:.2f} GB")
    print(f"Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    train_flows, val_flows, test_flows = load_data_parallel(
        folder_path, csv_path, seq_length, attack_funcs, n_processes,
    )
    
    end_time = time.time()
    end_memory = psutil.virtual_memory().used / (1024**3)  # GB
    
    print(f"\nData loading completed!")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Memory used: {end_memory - start_memory:.2f} GB")
    print(f"Final memory usage: {end_memory:.2f} GB")
    print(f"Train flows: {len(train_flows)}")
    print(f"Val flows: {len(val_flows)}")
    print(f"Test flows: {len(test_flows)}")
    
    return train_flows, val_flows, test_flows

# =====================================================
# TRAINING AND EVALUATION FUNCTIONS
# =====================================================

# Initialize scaler globally
scaler = GradScaler("cuda")

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for flow_a, flow_b, labels in dataloader:
        flow_a, flow_b, labels = flow_a.to(device), flow_b.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast(device_type=device.type):
            outputs = model(flow_a, flow_b)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for flow_a, flow_b, labels in dataloader:
            flow_a, flow_b, labels = flow_a.to(device), flow_b.to(device), labels.to(device)
            
            outputs = model(flow_a, flow_b)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', pos_label=1, zero_division=0
    )
    accuracy = accuracy_score(all_labels, all_predictions)
    
    cm = confusion_matrix(all_labels, all_predictions)
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
    return avg_loss, accuracy, precision, recall, f1, fpr

def train_binary_epoch(model, dataloader, optimizer, criterion, device):
    """Training function for DeepCoFFEA"""
    model.train()
    total_loss = 0
    for flow_a, flow_b, labels in dataloader:
        flow_a, flow_b, labels = flow_a.to(device), flow_b.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(flow_a, flow_b)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate_binary(model, dataloader, criterion, device):
    """Evaluation function for DeepCoFFEA"""
    model.eval()
    total_loss = 0
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for flow_a, flow_b, labels in dataloader:
            flow_a, flow_b, labels = flow_a.to(device), flow_b.to(device), labels.to(device)
            outputs = model(flow_a, flow_b)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary', zero_division=0)
    
    try:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions, labels=[0, 1]).ravel()
    except ValueError:
        return avg_loss, precision, recall, f1, 0.0
        
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return avg_loss, precision, recall, f1, fpr

# =====================================================
# EXPERIMENT RUNNER
# =====================================================

def run_single_experiment(experiment_name, train_flows, val_flows, test_flows, device, results_dict):
    """Run a single experiment with both models"""
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT: {experiment_name}")
    print(f"{'='*80}")
    
    # Create datasets
    train_dataset = FlowPairDataset(train_flows)
    val_dataset = FlowPairDataset(val_flows)
    test_dataset = FlowPairDataset(test_flows)
    
    # ===== TRANSFORMER MODEL =====
    print("\n--- Training CorrTransformer Model ---")
    
    # Create data loaders for transformer
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize transformer model
    model = FlowCorrelationTransformer(
        feature_dim=FEATURE_DIM,
        d_model=D_MODEL,
        n_head=N_HEAD,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        drop_path_rate=DROPOUT_PATH_RATE
    ).to(device)
    
    model = torch.compile(model, mode="default")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0 if epoch < 100 else 0.5 ** ((epoch - 100) // 50 + 1)
    )
    
    weight_for_class_1 = 1
    pos_weight_tensor = torch.tensor([weight_for_class_1], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    # Training loop for transformer
    best_val_f1 = -1
    best_epoch = -1
    best_model_state = None
    
    for epoch in range(TRANSFORMER_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_prec, val_recall, val_f1, val_fpr = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{TRANSFORMER_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val F1: {val_f1:.3f} (Prec: {val_prec:.3f}, Rec: {val_recall:.3f}, FPR: {val_fpr*100:.2f}%)")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
    
    # Test best transformer model
    print(f"\nBest validation F1 score: {best_val_f1:.4f} at epoch {best_epoch}")
    print("Testing best model on test set...")
    
    model.load_state_dict(best_model_state)
    test_loss, test_acc, test_prec, test_recall, test_f1, test_fpr = evaluate(model, test_loader, criterion, device)
    
    transformer_results = {
        'precision': test_prec,
        'recall': test_recall,
        'f1': test_f1,
        'fpr': test_fpr,
        'best_epoch': best_epoch
    }
    
    print(f"\nTransformer Test Results:")
    print(f"  Precision: {test_prec*100:.2f}%")
    print(f"  Recall: {test_recall*100:.2f}%")
    print(f"  F1-Score: {test_f1*100:.2f}%")
    print(f"  FPR: {test_fpr*100:.2f}%")
    
    # Clean up transformer model and data loaders
    del model, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    
    # Store results
    results_dict[experiment_name] = {
        'CorrTransformer': transformer_results
    }
    
    # Clean up DeepCoFFEA model and data loaders
    
    # Clean up datasets
    del train_dataset, val_dataset, test_dataset
    
    # Force garbage collection
    import gc
    gc.collect()
    torch.cuda.empty_cache()

# =====================================================
# MAIN EXECUTION
# =====================================================

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data paths
    folder_path = "../ProxyData/local_pcaps"
    csv_path = "../ProxyData/fullset_v2_info.csv"
    
    # Initialize data processor
    data_processor = DataProcessor("configs/background_distributions.json")
    
    # Load base datasets (with only bias removal)
    print("\n--- Loading Training Data ---")
    
    # Define experiments
    experiments = {
        'Exp5_IPDJitter': [data_processor.apply_ipd_jitter],
        'Exp1_BiasRemovalOnly': [],
    }
    
    # Results storage
    all_results = {}
    
    # Run each experiment
    for exp_name, attack_funcs in experiments.items():
        train_flows, val_flows, test_flows = timed_load_data(
            folder_path, 
            csv_path, 
            SEQ_LENGTH,
            attack_funcs, 
            n_processes=30  # Use 30 out of 32 cores
        )
        
        # Run experiment
        run_single_experiment(exp_name, train_flows, val_flows, test_flows, device, all_results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"experiment_results_{timestamp}.json"
        
        with open(results_filename, 'w') as f:
            json.dump(all_results, f, indent=4)
        
        # Clean up data to save memory
        del val_flows, test_flows
        
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"\nMemory cleaned up after {exp_name}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"experiment_results_{timestamp}.json"
    
    with open(results_filename, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\n\nResults saved to {results_filename}")
    
    # Print summary table
    print("\n\n" + "="*100)
    print("EXPERIMENT SUMMARY")
    print("="*100)
    print(f"{'Experiment':<25} {'Model':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'FPR (%)':<12}")
    print("-"*100)
    
    for exp_name, exp_results in all_results.items():
        for model_name, metrics in exp_results.items():
            print(f"{exp_name:<25} {model_name:<15} "
                  f"{metrics['precision']*100:<12.2f} "
                  f"{metrics['recall']*100:<12.2f} "
                  f"{metrics['f1']*100:<12.2f} "
                  f"{metrics['fpr']*100:<12.2f}")
    print("="*100)

if __name__ == "__main__":
    main()
