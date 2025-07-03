import pytest
import pandas as pd
import numpy as np

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
        sample_df, pkt_limit=6, attack_func_list=[processor.apply_attack]
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
        attack_func_list=[processor.apply_bias_removal, processor.apply_attack]
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