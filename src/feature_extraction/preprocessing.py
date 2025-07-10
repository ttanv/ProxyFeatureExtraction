"""
Module for dealing with prepossing before the final output. 
Used especially for doing the attacks and bias removal.
"""

import json
import logging
import numpy as np
import pandas as pd

class DataProcessor:
    """
    Encapsulates data modification logic and its required statistical parameters.
    
    The configuration (statistical parameters for attacks and bias removal)
    is loaded upon instantiation.
    """    
    def __init__(self, json_path='packet_lengths.json'):
        """Initializes statistical values upon initialization"""
        (
            self.empirical_pkt_lens, self.bg_len_mean, self.bg_len_std,
            self.bg_tg_mean, self.bg_tg_std
        ) = self._load_empirical_samples(json_path)
        
        return

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
        """
        Handles the bias removal steps:
        1) If the group has at least `pkt_limit` packets and 4th packet has pkt_len > 1300,
        drop the 4th and 6th packets.
        2) If a 4th packet still exists afterward, resample its pkt_len.
        """
        if len(group) > 3 and group.iloc[3]['pkt_len'] > 1300:
            group.drop(index=group.index[3], inplace=True)
            if len(group) > 4:
                group.drop(index=group.index[4], inplace=True)

        if len(group) > 3:
            if use_empirical_sampling and (self.empirical_pkt_lens is not None):
                new_length = np.random.choice(self.empirical_pkt_lens)
            else:
                if self.bg_len_mean is None or self.bg_len_std is None:
                    raise ValueError("Background distribution parameters are "
                                     "required when empirical sampling is disabled")
                mean_ = self.bg_len_mean
                std_ = self.bg_len_std
                new_length = np.random.normal(loc=mean_, scale=std_)
                new_length = max(1, int(round(new_length)))

            group.at[group.index[3], 'pkt_len'] = new_length

        return group

    def apply_decorrelation_attack(self, group):
        """
        Applies timing attack modification to the group
        """
        if len(group) > 3:
            if self.bg_tg_mean is None or self.bg_tg_std is None:
                raise ValueError("Background timing distribution parameters are required for decorrelation attack.")
            
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
    
    
    def apply_targeted_padding(self, group, n_packets_to_pad, pad_size):
        """
        Applies padding to the first `n_packets_to_pad` packets in the flow.
        A random amount of padding between 1 and `pad_size` is added.
        """
        # Determine the number of packets to pad (up to the limit or flow length)
        num_to_pad = min(len(group), n_packets_to_pad)
        
        if num_to_pad == 0:
            return group

        # Get the indices of the first `num_to_pad` packets
        target_indices = group.index[:num_to_pad]
        
        # Generate random padding values for each target packet
        padding_values = np.random.randint(1, pad_size + 1, size=num_to_pad)
        
        # Add the padding to the 'pkt_len' of the targeted packets
        group.loc[target_indices, 'pkt_len'] += padding_values
        
        return group
    
    def apply_ipd_jitter(self, group, n_packets_to_jitter, max_delay_s):
        """
        Applies random timing jitter between the first `n_packets_to_jitter` packets.
        """
        # Determine how many packets' timestamps we will adjust
        num_to_jitter = min(len(group) -1, n_packets_to_jitter)
        
        if num_to_jitter <= 0:
            return group

        # Iterate through the packets and apply a delay that affects all subsequent packets
        for i in range(num_to_jitter):
            # Generate a random delay for the current packet
            jitter = np.random.uniform(0, max_delay_s)
            
            # Get the index of the packet *after which* the delay will be added
            packet_idx = group.index[i]
            
            # Apply the delay to all subsequent packets' timestamps
            mask = group.index > packet_idx
            group.loc[mask, 'ts_relative'] += jitter
            
        return group
    
    
    def apply_packet_reshaping(self, group: pd.DataFrame,
                           split_threshold: int = 1000,
                           max_splits: int = 3,
                           min_pkt_size: int = 128) -> pd.DataFrame:
        """
        Structural obfuscation: split oversized packets into several
        random-length segments, preserving total bytes and order.

        Parameters
        ----------
        group : pd.DataFrame
            One bidirectional flow with columns ['pkt_len', 'ts_relative', ...].
        split_threshold : int
            Packets larger than this are candidates for splitting (bytes).
        max_splits : int
            Upper bound on how many pieces a single packet may become.
        min_pkt_size : int
            Smallest allowed segment size (bytes).

        Returns
        -------
        pd.DataFrame
            New flow with reshaped packet lengths and timestamps.
        """
        new_rows = []
        indices_to_drop = []

        for idx, row in group.iterrows():
            plen = row['pkt_len']
            if plen > split_threshold:
                indices_to_drop.append(idx)
                ts0 = row['ts_relative']

                # choose 2 … max_splits pieces
                k = np.random.randint(2, max_splits + 1)

                # random Dirichlet split, enforce minimum size
                shares = np.random.dirichlet(np.ones(k))
                seg_lengths = np.maximum(
                    np.round(shares * (plen - k * min_pkt_size)).astype(int) + min_pkt_size,
                    min_pkt_size
                )
                seg_lengths[-1] = plen - seg_lengths[:-1].sum()  # byte conservation

                for i, slen in enumerate(seg_lengths):
                    new_row = row.copy()
                    new_row['pkt_len'] = int(slen)
                    new_row['ts_relative'] = ts0 + i * 1e-6  # preserve order
                    new_rows.append(new_row)
            else:
                new_rows.append(row)

        reshaped = pd.DataFrame(new_rows).sort_values('ts_relative').reset_index(drop=True)
        return reshaped


    def apply_changes(self, df, pkt_limit, attack_func_list):
        """
        Main function that applies attacks from attack_func_list sequentially
        """
        updated_groups = []

        if df.empty:
            return df

        for _, group in df.groupby('conn', sort=False):
            if len(group) >= pkt_limit:
                temp_group = group.sort_values(by='ts_relative', ascending=True).copy()
                
                # Apply bias removal
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