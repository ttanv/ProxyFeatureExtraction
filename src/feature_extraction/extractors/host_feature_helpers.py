"""
Created on Wed Nov 27 14:55:46 2024

@author: mounarabhi
"""


import pandas as pd
import pdb
import numpy as np


def calc_seconds(ts_list):
    return ts_list[-1] - ts_list[0]

def extract_features_by_conn(df, gw=True, max_pkts=20, comp_pkts_limit=50, 
                             bias_fix = False, attack = False, fix_dict=None):
    
    if bias_fix:
        assert fix_dict is not None, "fix_dict must be provided when bias_fix is True"
        for conn_id, conn_data in df.groupby('conn'):
        # Sort and drop packets
            conn_data = conn_data.sort_values('ts_relative')
            if len(conn_data) > 3 and conn_data.iloc[3]['pkt_len'] > 1300:
                conn_data = conn_data.drop(conn_data.index[3]).reset_index(drop=True)
                conn_data = conn_data.drop(conn_data.index[4]).reset_index(drop=True)
                new_length = np.random.choice(fix_dict['empirical_packet_lengths'])
                conn_data.at[3, 'pkt_len'] = new_length

            if attack:

                conn_data.reset_index(drop=True, inplace=True)
                new_timing = np.random.lognormal(
                    mean=fix_dict['timing_distribution']['mean'],
                    sigma=fix_dict['timing_distribution']['std']
                )
            
                # Calculate current timing difference
                old_timing = conn_data.iloc[3]['ts_relative'] - conn_data.iloc[2]['ts_relative']
            
                # Calculate and apply timing adjustment
                timing_adjustment = old_timing - new_timing 
                conn_data.loc[3:, 'ts_relative'] = conn_data.loc[3:, 'ts_relative'] - timing_adjustment
            df.drop(df[df['conn'] == conn_id].index, inplace=True)
            df = pd.concat([df, conn_data], ignore_index=True)

    rows_before = len(df)

    # Perform the dropna operation
    df = df.dropna(subset=['ts_relative', 'pkt_len', 'conn'])

    # Calculate and print the number of rows dropped
    rows_after = len(df)
    rows_dropped = rows_before - rows_after
    print(f"Number of rows before dropna: {rows_before}")
    print(f"Number of rows after dropna: {rows_after}")
    print(f"Total rows dropped: {rows_dropped}")
    if rows_dropped > 0:
        pdb.set_trace()
    grouped = df.groupby('conn')

    all_features = []
    group_start_times = []

    for conn_name, group in grouped:
        if len(group) < max_pkts:
            continue
        group2 = group
        # head() value should take min of comp_pkts_limit and len(group)
        group = group.head(min(comp_pkts_limit, len(group)))

        features = {'conn': conn_name}
        group = group.sort_values(by='ts_relative')

        # Total time for the connection
        total_time = calc_seconds(group['ts_relative'].tolist())
        features['pkts_rate'] = len(group) / total_time if total_time > 0 else 0

        # Duration analysis
        first_ts = group2['ts_relative'].iloc[0]
        last_ts = group2['ts_relative'].iloc[-1]
        durations = last_ts - first_ts
        features['duration'] = durations
        # Time gaps between connections
        group_start_times.append(first_ts)  # First timestamp in the group

        # Volume features (including all packets)
        total_pkts = group['pkt_len']
        features['mean_vol_total_pkts'] = total_pkts.mean() if not total_pkts.empty else 0
        features['median_vol_total_pkts'] = total_pkts.median() if not total_pkts.empty else 0
        features['mode_vol_total_pkts'] = total_pkts.mode()[0] if not total_pkts.empty else 0
        features['std_vol_total_pkts'] = total_pkts.std() if not total_pkts.empty else 0
        features['skew_vol_total_pkts'] = total_pkts.skew() if not total_pkts.empty else 0
        features['kurtosis_vol_total_pkts'] = total_pkts.kurtosis() if not total_pkts.empty else 0

        # Filter by source and destination IP addresses (gateway scenario)
        if gw:
            group_sent = group[(group['dst_ip'] == '10.0.2.16') | (group['dst_ip'] == '10.0.2.15')]
            group_recv = group[(group['src_ip'] == '10.0.2.16') | (group['src_ip'] == '10.0.2.15')]
        else:
            group_recv = group[(group['dst_ip'] == '10.0.2.16') | (group['dst_ip'] == '10.0.2.15')]
            group_sent = group[(group['src_ip'] == '10.0.2.16') | (group['src_ip'] == '10.0.2.15')]

        # Calculate sent packet statistics
        total_sent = group_sent['pkt_len']
        features['mean_bytes_sent'] = total_sent.mean() if not total_sent.empty else 0
        features['median_bytes_sent'] = total_sent.median() if not total_sent.empty else 0
        features['mode_bytes_sent'] = total_sent.mode()[0] if not total_sent.empty else 0
        features['std_bytes_sent'] = total_sent.std() if not total_sent.empty else 0
        features['skew_bytes_sent'] = total_sent.skew() if not total_sent.empty else 0
        features['kurtosis_bytes_sent'] = total_sent.kurtosis() if not total_sent.empty else 0

        # Calculate received packet statistics
        total_recv = group_recv['pkt_len']
        features['mean_bytes_recv'] = total_recv.mean() if not total_recv.empty else 0
        features['median_bytes_recv'] = total_recv.median() if not total_recv.empty else 0
        features['mode_bytes_recv'] = total_recv.mode()[0] if not total_recv.empty else 0
        features['std_bytes_recv'] = total_recv.std() if not total_recv.empty else 0
        features['skew_bytes_recv'] = total_recv.skew() if not total_recv.empty else 0
        features['kurtosis_bytes_recv'] = total_recv.kurtosis() if not total_recv.empty else 0

        all_features.append(features)

    # Calculate time gaps between connection groups
    group_start_times = sorted(group_start_times)
    time_diffs = [abs(group_start_times[i + 1] - group_start_times[i]) for i in range(len(group_start_times) - 1)]

    # Add avg_time_gap 
    for i, features in enumerate(all_features[:-1]):  
        features['gap_between_conns'] = time_diffs[i]  
    if all_features:
        all_features[-1]['gap_between_conns'] = 0  # Set last connection's gap to 0

    # Convert the list of dictionaries to a DataFrame for easier analysis
    features_df = pd.DataFrame(all_features)
    return features_df


        