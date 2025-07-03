import pytest
import pandas as pd
import numpy as np
import cudf
import cupy as cp

# Assuming the class is saved in the structure:

from feature_extraction.extractors.corr_extractor import CorrFeatureExtractor

# Sequential (CPU-based) implementation for correctness comparison
def get_correlation_array_sequential(df, gateway_df, pkt_limit, bin_size_seconds):
    """
    A sequential, pandas/numpy-based implementation of the correlation logic
    for verification purposes.
    """
    # 1. Filter connections with less than pkt_limit packets
    group_sizes = df.groupby('conn').size()
    valid_conns = group_sizes[group_sizes >= pkt_limit].index
    df_filtered = df[df['conn'].isin(valid_conns)].copy()

    # 2. For each valid connection, keep only the first pkt_limit packets
    df_limited = df_filtered.groupby('conn').head(pkt_limit).copy()

    # 3. Binning
    bin_factor = 1.0 / bin_size_seconds
    df_limited.loc[:, 'time_bin'] = np.floor(df_limited['ts_relative'] * bin_factor) / bin_factor
    gateway_df.loc[:, 'time_bin'] = np.floor(gateway_df['ts_relative'] * bin_factor) / bin_factor

    # 4. Group by bins
    df_binned = df_limited.groupby(['conn', 'time_bin'])['pkt_len'].sum().reset_index()
    gateway_binned = gateway_df.groupby('time_bin')['pkt_len'].sum().reset_index()

    results = []

    for conn_val in df_limited['conn'].unique():
        conn_subset = df_limited[df_limited['conn'] == conn_val]
        start_time = conn_subset['ts_relative'].min()
        end_time = conn_subset['ts_relative'].max() + 1.0

        # Filter gateway traffic for the connection's time range
        gateway_sub = gateway_binned[
            (gateway_binned['time_bin'] >= start_time) &
            (gateway_binned['time_bin'] <= end_time)
        ].copy()

        conn_bins = df_binned[df_binned['conn'] == conn_val].copy()

        if gateway_sub.empty:
            # Handle cases with no overlapping gateway traffic
            metrics = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        else:
            # Merge and compute correlation
            merged = pd.merge(gateway_sub, conn_bins, on='time_bin', how='outer').fillna(0)
            gw_vals = merged['pkt_len_x'].values
            rl_vals = merged['pkt_len_y'].values

            gw_mean, gw_std = np.mean(gw_vals), np.std(gw_vals) + 1e-9
            rl_mean, rl_std = np.mean(rl_vals), np.std(rl_vals) + 1e-9

            gw_z = (gw_vals - gw_mean) / gw_std
            rl_z = (rl_vals - rl_mean) / rl_std

            corr_array = gw_z * rl_z

            # Get metrics using numpy
            corr_count = int(corr_array.size)
            corr_sum = float(np.sum(corr_array))
            corr_mean = float(np.mean(corr_array))
            corr_median = float(np.median(corr_array))
            corr_minimum = float(np.min(corr_array))
            corr_maximum = float(np.max(corr_array))
            corr_range = corr_maximum - corr_minimum
            corr_variance = float(np.var(corr_array))
            corr_std_dev = float(np.std(corr_array))

            metrics = (
                corr_count, corr_sum, corr_mean,
                corr_median, corr_minimum, corr_maximum,
                corr_range, corr_variance, corr_std_dev
            )

        results.append((conn_val, *metrics))

    columns = [
        'conn', 'corr_count', 'corr_sum', 'corr_mean', 'corr_median',
        'corr_minimum', 'corr_maximum', 'corr_range', 'corr_variance', 'corr_std_dev'
    ]
    return pd.DataFrame(results, columns=columns)


@pytest.fixture
def sample_data():
    """Pytest fixture to create sample DataFrames for testing."""
    # Connection DataFrame
    conn_data = {
        'conn': ['conn1'] * 25 + ['conn2'] * 30 + ['conn3'] * 15,
        'ts_relative': np.concatenate([
            np.linspace(0, 4.8, 25),
            np.linspace(5, 10.8, 30),
            np.linspace(11, 13.8, 15)
        ]),
        'pkt_len': np.random.randint(100, 1500, 70)
    }
    conn_df = pd.DataFrame(conn_data)

    # Gateway DataFrame
    gateway_data = {
        'ts_relative': np.linspace(0, 15, 150),
        'pkt_len': np.random.randint(500, 3000, 150)
    }
    gateway_df = pd.DataFrame(gateway_data)

    return cudf.from_pandas(conn_df), cudf.from_pandas(gateway_df)

def test_smoke_and_no_errors(sample_data):
    """
    Test that the feature extractor runs without raising any exceptions.
    """
    conn_df, gateway_df = sample_data
    extractor = CorrFeatureExtractor(conn_df=conn_df, gateway_df=gateway_df)
    result_df = extractor.process_df(pkt_limit=20)

    assert isinstance(result_df, pd.DataFrame)
    assert not result_df.empty
    expected_columns = [
        'conn', 'corr_count', 'corr_sum', 'corr_mean', 'corr_median',
        'corr_minimum', 'corr_maximum', 'corr_range', 'corr_variance', 'corr_std_dev'
    ]
    assert all(col in result_df.columns for col in expected_columns)
    # conn3 should be filtered out as it has less than 20 packets
    assert 'conn3' not in result_df['conn'].values

def test_correctness_against_sequential(sample_data):
    """
    Test that the GPU implementation matches the sequential CPU implementation.
    """
    conn_cudf, gateway_cudf = sample_data
    pkt_limit = 20
    bin_size = 0.1

    # Get result from the GPU implementation
    extractor = CorrFeatureExtractor(conn_df=conn_cudf, gateway_df=gateway_cudf)
    gpu_result = extractor.process_df(pkt_limit=pkt_limit).sort_values('conn').reset_index(drop=True)

    # Get result from the sequential CPU implementation
    conn_df_pd = conn_cudf.to_pandas()
    gateway_df_pd = gateway_cudf.to_pandas()
    
    # Note: The user's `process_df` calls `_get_correlation_array(self.gateway_df, self.conn_df, ...)`
    # This seems like a typo and that it should be `(self.conn_df, self.gateway_df, ...)`
    # We will test against the code as written.
    cpu_result = get_correlation_array_sequential(
        conn_df_pd, gateway_df_pd, pkt_limit, bin_size
    ).sort_values('conn').reset_index(drop=True)

    # Compare the two DataFrames
    pd.testing.assert_frame_equal(gpu_result, cpu_result, check_dtype=True, atol=1e-5)

def test_empty_input():
    """
    Test the behavior with empty input DataFrames.
    """
    conn_df = cudf.DataFrame({'conn': [], 'ts_relative': [], 'pkt_len': []})
    gateway_df = cudf.DataFrame({'ts_relative': [], 'pkt_len': []})
    
    extractor = CorrFeatureExtractor(conn_df=conn_df, gateway_df=gateway_df)
    result_df = extractor.process_df(pkt_limit=20)

    assert isinstance(result_df, pd.DataFrame)
    assert result_df.empty

def test_no_overlapping_traffic():
    """
    Test the case where connection and gateway traffic do not overlap in time.
    """
    conn_data = {
        'conn': ['conn1'] * 25,
        'ts_relative': np.linspace(0, 2, 25),
        'pkt_len': np.random.randint(100, 200, 25)
    }
    conn_df = cudf.from_pandas(pd.DataFrame(conn_data))

    gateway_data = {
        'ts_relative': np.linspace(100, 102, 50),
        'pkt_len': np.random.randint(1000, 1500, 50)
    }
    gateway_df = cudf.from_pandas(pd.DataFrame(gateway_data))

    extractor = CorrFeatureExtractor(conn_df=conn_df, gateway_df=gateway_df)
    result_df = extractor.process_df(pkt_limit=20)

    assert not result_df.empty
    # Expect zero metrics as there is no overlapping gateway traffic
    for col in result_df.columns:
        if col != 'conn':
            assert np.allclose(result_df[col], 0)