import pytest
import pandas as pd
import numpy as np
from functools import partial

from feature_extraction.preprocessing import DataProcessor

@pytest.fixture
def processor(monkeypatch):
    # Patch _load_empirical_samples to avoid file I/O
    monkeypatch.setattr(
        DataProcessor, "_load_empirical_samples",
        lambda self, _: ([100, 200, 300], 150, 10, 0.5, 0.1)
    )
    return DataProcessor(json_path="dummy.json")

@pytest.fixture
def sample_df():
    # Create a DataFrame with two connections, each with 6 packets
    data = []
    for conn_id in [1, 2]:
        for i in range(6):
            data.append({
                "conn": conn_id,
                "pkt_len": 1400 if i == 3 else 100 + i,
                "ts_relative": float(i)
            })
    return pd.DataFrame(data)

def test_apply_changes_bias_removal_only(processor, sample_df):
    # Only bias removal
    out = processor.apply_changes(
        sample_df, pkt_limit=6, attack_func_list=[processor.apply_bias_removal]
    )
    # 4th and 6th packets should be dropped for each group
    for conn_id in [1, 2]:
        group = out[out["conn"] == conn_id]
        assert len(group) == 4
        # 4th packet (index 3) should have new pkt_len from empirical
        assert group.iloc[3]["pkt_len"] in [100, 200, 300]

def test_apply_changes_attack_only(processor, sample_df, monkeypatch):
    # Patch np.random.lognormal to return a fixed value
    monkeypatch.setattr(np.random, "lognormal", lambda mean, sigma: 0.5)
    # Only attack
    out = processor.apply_changes(
        sample_df, pkt_limit=6, attack_func_list=[processor.apply_decorrelation_attack]
    )
    # No rows dropped, but ts_relative for packets >= 3 should be shifted
    for conn_id in [1, 2]:
        group = out[out["conn"] == conn_id].sort_values("ts_relative")
        # The difference between 4th and 3rd packet should be 0.5
        idx = group.index[3]
        prev_idx = group.index[2]
        diff = group.loc[idx, "ts_relative"] - group.loc[prev_idx, "ts_relative"]
        assert abs(diff - 0.5) < 1e-6

def test_apply_changes_both_funcs(processor, sample_df, monkeypatch):
    # Patch np.random.lognormal to return a fixed value
    monkeypatch.setattr(np.random, "lognormal", lambda mean, sigma: 0.5)
    # Both bias removal and attack
    out = processor.apply_changes(
        sample_df, pkt_limit=6,
        attack_func_list=[processor.apply_bias_removal, processor.apply_decorrelation_attack]
    )
    # 4th and 6th packets dropped, so only 4 packets per group
    for conn_id in [1, 2]:
        group = out[out["conn"] == conn_id]
        assert len(group) == 4
        # ts_relative for last packet should be shifted
        if len(group) > 3:
            idx = group.index[3]
            prev_idx = group.index[2]
            diff = group.loc[idx, "ts_relative"] - group.loc[prev_idx, "ts_relative"]
            assert abs(diff - 0.5) < 1e-6

def test_apply_changes_no_attack(processor, sample_df):
    # No attack functions
    out = processor.apply_changes(
        sample_df, pkt_limit=6, attack_func_list=[]
    )
    # Should be unchanged
    pd.testing.assert_frame_equal(
        out.sort_index(axis=1), sample_df.sort_index(axis=1)
    )

def test_apply_changes_pkt_limit(processor, sample_df):
    # pkt_limit higher than group size, so no changes
    out = processor.apply_changes(
        sample_df, pkt_limit=10, attack_func_list=[processor.apply_bias_removal]
    )
    pd.testing.assert_frame_equal(
        out.sort_index(axis=1), sample_df.sort_index(axis=1)
    )

def test_apply_targeted_padding(processor, sample_df):
    """Test for apply_targeted_padding"""
    np.random.seed(42)
    n_packets_to_pad = 3
    pad_size = 10
    
    padding_attack = partial(processor.apply_targeted_padding, 
                             n_packets_to_pad=n_packets_to_pad, 
                             pad_size=pad_size)

    out = processor.apply_changes(
        sample_df.copy(), pkt_limit=6, attack_func_list=[padding_attack]
    )

    for conn_id in [1, 2]:
        group = out[out["conn"] == conn_id]
        original_group = sample_df[sample_df["conn"] == conn_id]
        
        # Check first n packets are padded
        for i in range(n_packets_to_pad):
            assert group.iloc[i]['pkt_len'] > original_group.iloc[i]['pkt_len']
            assert group.iloc[i]['pkt_len'] <= original_group.iloc[i]['pkt_len'] + pad_size

        # Check other packets are not padded
        for i in range(n_packets_to_pad, len(group)):
            assert group.iloc[i]['pkt_len'] == original_group.iloc[i]['pkt_len']

def test_apply_ipd_jitter(processor, sample_df, monkeypatch):
    """Test for apply_ipd_jitter"""
    n_packets_to_jitter = 3
    max_delay_s = 0.1
    
    # Mock random jitter to be a constant value
    jitter_val = 0.05
    monkeypatch.setattr(np.random, "uniform", lambda low, high: jitter_val)
    
    jitter_attack = partial(processor.apply_ipd_jitter, 
                            n_packets_to_jitter=n_packets_to_jitter, 
                            max_delay_s=max_delay_s)

    out = processor.apply_changes(
        sample_df.copy(), pkt_limit=6, attack_func_list=[jitter_attack]
    )

    for conn_id in [1, 2]:
        group = out[out["conn"] == conn_id]
        original_group = sample_df[sample_df["conn"] == conn_id]
        
        # first packet is unchanged
        assert group.iloc[0]['ts_relative'] == original_group.iloc[0]['ts_relative']
        
        # subsequent packets have accumulated jitter
        for i in range(1, len(group)):
            num_jitters_applied = min(i, n_packets_to_jitter)
            expected_jitter = num_jitters_applied * jitter_val
            expected_ts = original_group.iloc[i]['ts_relative'] + expected_jitter
            assert abs(group.iloc[i]['ts_relative'] - expected_ts) < 1e-9

def test_apply_packet_reshaping(processor, sample_df):
    """Test for apply_packet_reshaping"""
    np.random.seed(39)
    reshaping_attack = partial(processor.apply_packet_reshaping, 
                               split_threshold=1300, 
                               max_splits=3, 
                               min_pkt_size=32)

    out = processor.apply_changes(
        sample_df.copy(), pkt_limit=6, attack_func_list=[reshaping_attack]
    )

    for conn_id in [1, 2]:
        group = out[out["conn"] == conn_id]
        original_group = sample_df[sample_df["conn"] == conn_id]

        # Total bytes should be conserved
        assert group['pkt_len'].sum() == original_group['pkt_len'].sum()
        
        # One packet was > 1300, so it should be split, resulting in more packets
        assert len(group) > len(original_group)
        
        # Check that no packet is larger than the split threshold
        assert group['pkt_len'].max() <= 1300
        
        # Check that timestamps are close to original and ordered
        original_split_pkt_ts = original_group.iloc[3]['ts_relative']
        split_packets = group[
            (group['ts_relative'] >= original_split_pkt_ts) &
            (group['ts_relative'] < original_split_pkt_ts + 1)
        ]
        assert np.all(np.diff(split_packets['ts_relative']) > 0)