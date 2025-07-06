"""
Feature extractor implementation for the cosine similarity-based correlation features
"""

# TODO: Ensure that the output df has the correct labels for each row

import pandas as pd
import cudf
import cupy as cp
from feature_extraction.extractors.base_extractor import BaseFeatureExtractor

BIN_SIZE_SECONDS = 0.1

class CorrFeatureExtractor(BaseFeatureExtractor):
    """
    Class for the implementation of the correlation feature
    """
    def __init__(self, conn_df, gateway_df):
        if gateway_df is None or conn_df is None:
            raise ValueError("Gateway df and connection df must be present for correlation based extraction")
        
        super().__init__("corr_feature", conn_df, gateway_df)
    
    def _get_zero_metrics(self):
        """
        Returns zero for each metric, used for null cases
        """
        return (0, 0, 0, 
                0, 0, 0,
                0, 0, 0)
        
    def _get_metrics(self, array_cp):
        """
        Calculates statistical metrics for the given array in cupy
        """
        corr_count = int(array_cp.size)
        corr_sum = float(cp.sum(array_cp))
        corr_mean = float(cp.mean(array_cp))
        corr_median = float(cp.median(array_cp))
        corr_minimum = float(cp.min(array_cp))
        corr_maximum = float(cp.max(array_cp))
        corr_range = corr_maximum - corr_minimum
        corr_variance = float(cp.var(array_cp))
        corr_std_dev = float(cp.std(array_cp))
        
        return (
            corr_count, corr_sum, corr_mean, 
            corr_median, corr_minimum, corr_maximum, 
            corr_range, corr_variance, corr_std_dev
        )
    
    
    def _process_single_connection(self, conn_val, df_binned, gateway_binned, 
                                   gateway_time_bins, minmax_map):
        """
        Process a single connection for correlation analysis
        """
        
        # Retrieve min_time, max_time
        times = minmax_map.get(conn_val)
        if not times:
            return None
        tmin, tmax = times
        if cp.isnan(tmin) or cp.isnan(tmax):
            return None

        start_time = tmin
        end_time = tmax + 1.0  # bound_range hardcoded to 1.0

        # Subset df_binned for this connection
        sub_mask = (df_binned['conn'] == conn_val)
        conn_binned = df_binned[sub_mask]
        if conn_binned.empty:
            return None

        # Slice gateway bins using searchsorted
        # Make them 1D CuPy arrays, same dtype as gateway_time_bins
        start_time_cp = cp.asarray([start_time], dtype=gateway_time_bins.dtype)
        end_time_cp = cp.asarray([end_time], dtype=gateway_time_bins.dtype)

        left_idx_arr = cp.searchsorted(gateway_time_bins, start_time_cp, side='left')
        right_idx_arr = cp.searchsorted(gateway_time_bins, end_time_cp, side='right')

        # left_idx_arr and right_idx_arr are 1D Cupy arrays. Extract the int index:
        left_idx = int(left_idx_arr[0].item())
        right_idx = int(right_idx_arr[0].item())

        gateway_sub = gateway_binned.iloc[left_idx:right_idx]
        if gateway_sub.empty:
            return (conn_val, *self._get_zero_metrics())

        # Rename columns for clarity
        gateway_sub = gateway_sub.rename(columns={'pkt_len': 'gw_len'})
        conn_binned = conn_binned.rename(columns={'pkt_len': 'rl_len'})

        # Merge on time_bin
        merged = gateway_sub.merge(conn_binned, on='time_bin', how='outer').fillna({'gw_len':0, 'rl_len':0})

        # Convert to CuPy arrays
        gw_vals = merged['gw_len'].values
        rl_vals = merged['rl_len'].values

        # Compute z-scores on GPU
        gw_mean, gw_std = cp.mean(gw_vals), cp.std(gw_vals) + 1e-9
        rl_mean, rl_std = cp.mean(rl_vals), cp.std(rl_vals) + 1e-9

        gw_z = (gw_vals - gw_mean) / gw_std
        rl_z = (rl_vals - rl_mean) / rl_std

        # Element-wise correlation array
        corr_array = gw_z * rl_z

        # Get metrics
        metrics = self._get_metrics(corr_array)
        return (conn_val, *metrics)
    
    
    def _get_correlation_array(self, df, gateway_df, pkt_limit):
        """
        GPU-accelerated approach using cuDF & CuPy.
        Considers all connections with >20 packets but analyzes only first 20 packets.
        """

        # 1) Convert to cuDF if needed
        if not isinstance(gateway_df, cudf.DataFrame):
            gateway_gdf = cudf.DataFrame.from_pandas(gateway_df)
        else:
            gateway_gdf = gateway_df

        if not isinstance(df, cudf.DataFrame):
            df_gdf = cudf.DataFrame.from_pandas(df)
        else:
            df_gdf = df

        # 2) Ensure numeric columns & drop NA
        gateway_gdf['ts_relative'] = gateway_gdf['ts_relative'].astype(float)
        gateway_gdf['pkt_len'] = gateway_gdf['pkt_len'].astype(float)
        gateway_gdf = gateway_gdf.dropna(subset=['ts_relative', 'pkt_len'])

        df_gdf['ts_relative'] = df_gdf['ts_relative'].astype(float)
        df_gdf['pkt_len'] = df_gdf['pkt_len'].astype(float)
        df_gdf = df_gdf.dropna(subset=['ts_relative', 'pkt_len'])

        # 3) Sort and identify valid connections
        gateway_gdf = gateway_gdf.sort_values('ts_relative')
        df_gdf = df_gdf.sort_values(['conn', 'ts_relative'])
        
        # First identify connections that have more than 20 packets
        group_sizes = df_gdf.groupby('conn').size().reset_index(name='group_count')
        valid_conns = group_sizes[group_sizes['group_count'] >= 20]['conn']
        
        # Filter to keep only valid connections
        df_gdf = df_gdf.merge(valid_conns.to_frame('conn'), on='conn')
        
        # Now for each valid connection, keep only first 20 packets for correlation
        df_gdf = df_gdf.sort_values(['conn', 'ts_relative'])
        df_gdf['row_num'] = df_gdf.groupby('conn').cumcount()
        df_gdf = df_gdf[df_gdf['row_num'] < pkt_limit].drop(columns=['row_num'])

        # 4) Bin entire data once
        bin_factor = 1.0 / BIN_SIZE_SECONDS
        gateway_gdf['time_bin'] = cp.floor(gateway_gdf['ts_relative'] * bin_factor) / bin_factor
        df_gdf['time_bin'] = cp.floor(df_gdf['ts_relative'] * bin_factor) / bin_factor

        # 5) Summation grouped by bin
        gateway_binned = gateway_gdf.groupby('time_bin')['pkt_len'].sum().reset_index()
        gateway_binned = gateway_binned.sort_values('time_bin')  # ensure ascending

        # Summation grouped by (conn, time_bin)
        df_binned = df_gdf.groupby(['conn', 'time_bin'])['pkt_len'].sum().reset_index()

        # 6) Prepare search-slice for gateway bins
        gateway_time_bins = gateway_binned['time_bin'].values  # GPU array

        # 7) Find min/max for each conn (returns a DataFrame with columns like "ts_relative_min", "ts_relative_max")
        minmax_df = df_gdf.groupby('conn')['ts_relative'].agg(['min', 'max']).reset_index()

        # 8) Build a dictionary on CPU
        # Convert the group result to Pandas or host arrays
        # If minmax_df is not huge, converting to_pandas() is fine:
        mm_pdf = minmax_df.to_pandas()  # small DataFrame
        minmax_map = {}
        for _, row in mm_pdf.iterrows():
            conn_key = row['conn']
            tmin = row['min']
            tmax = row['max']
            minmax_map[conn_key] = (tmin, tmax)

        # 9) Unique connections
        unique_conns = df_gdf['conn'].unique()

        # 10) Process each connection in a Python loop
        results = []
        for conn_val in unique_conns.to_pandas():
            result_df = self._process_single_connection(conn_val, df_binned, gateway_binned, gateway_time_bins, minmax_map)
            if result_df:
                results.append(result_df)

        # Convert final results to Pandas
        columns = [
            'conn', 'corr_count', 'corr_sum', 'corr_mean', 'corr_median',
            'corr_minimum', 'corr_maximum', 'corr_range', 'corr_variance', 'corr_std_dev'
        ]
        
        result_df = pd.DataFrame(results, columns=columns)
        return result_df
    
    
    def process_df(self, pkt_limit):
        """
        Returns a df with corr features for each connection
        """
        return self._get_correlation_array(self.conn_df, self.gateway_df, pkt_limit)