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

    def apply_attack(self, group):
        """
        Applies timing attack modification to the group
        """
        if len(group) > 3:
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