"""
Feature extractor implementation of website fingerprinting Traffic Analysis (TA) features
"""

import pandas as pd
from feature_extraction.extractors.base_extractor import BaseFeatureExtractor
from feature_extraction.extractors.hayes_usenix2019_features import get_features
from feature_extraction.extractors.host_feature_helpers import extract_features_by_conn


class TAFeatureExtractor(BaseFeatureExtractor):
    """
    Class for extracting traffic analysis features called TA in the paper
    """
    def __init__(self, conn_df):
        super().__init__("ta_features", conn_df)
        if conn_df is None:
            raise ValueError("Connection df must be given as argument")
        
        # Feature names to be used as column name
        self.feature_names = [
            "max_in", "max_out", "max_total", "avg_in", "avg_out", "avg_total",
            "std_in", "std_out", "std_total", "75th_percentile_in", "75th_percentile_out", "75th_percentile_total",
            "25th_percentile_in_time", "50th_percentile_in_time", "75th_percentile_in_time", "100th_percentile_in_time",
            "25th_percentile_out_time", "50th_percentile_out_time", "75th_percentile_out_time", "100th_percentile_out_time",
            "25th_percentile_total_time", "50th_percentile_total_time", "75th_percentile_total_time", "100th_percentile_total_time",
            "nb_pkts_in", "nb_pkts_out", "nb_pkts_total",
            "nb_pkts_in_f30", "nb_pkts_out_f30", "nb_pkts_in_l30", "nb_pkts_out_l30",
            "std_pkt_conc_out20", "avg_pkt_conc_out20", "avg_per_sec", "std_per_sec", "avg_order_in",
            "avg_order_out", "std_order_in", "std_order_out", "medconc", "med_per_sec", "min_per_sec",
            "max_per_sec", "maxconc", "perc_in", "perc_out", "sum_altconc", "sum_alt_per_sec",
            "sum_number_pkts", "sum_intertimestats"
        ]
        self.feature_names.extend([f"altconc_{i+1}" for i in range(20)])
        self.feature_names.extend([f"alt_per_sec_{i+1}" for i in range(20)])
        self.feature_names.extend([f"conc_{i+1}" for i in range(60)])
        
        
    def process_df(self, pkt_limit):
        """Process a single CSV file and extract features using pandas for efficiency."""    
        features_data = []
        for conn_name, group in self.conn_df.groupby("conn", sort=False):
            conn_pkts_list = group.values.tolist() # Convert group to list of lists
            
            if len(conn_pkts_list) < pkt_limit:
                continue

            conn_features_values = get_features(conn_pkts_list[:pkt_limit], conn_name, limit=0)
            if conn_features_values:
                features_data.append({'conn': conn_name, **dict(zip(self.feature_names, conn_features_values))})
        
        if not features_data:
            return None

        # Extract additional features
        features_k_final = pd.DataFrame(features_data)
        features_host = extract_features_by_conn(self.conn_df, gw=False) # This function is external
        
        if 'conn' in features_host.columns and not features_k_final.empty:
            merged_df = pd.merge(features_host, features_k_final, on='conn', how='inner')
            # TODO: Fix this labeling thing, the label should be give in script file
            # merged_df['pcap_nb'] = Path(folder_path).name # More robust way to get folder name
            # merged_df['label'] = int(prefix == "relayed")
            return merged_df

        return None
    
    
    
    
