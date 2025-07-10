import pytest
import pandas as pd
import numpy as np
from feature_extraction.extractors.slt_extractor import SLTExtractor, PacketFeatures, ConnectionFeatures

@pytest.fixture
def sample_data():
    """Pytest fixture to create a sample DataFrame for testing."""
    conn_data = {
        'conn': ['conn1'] * 25 + ['conn2'] * 30,
        'ts_relative': np.concatenate([
            np.linspace(0, 4.8, 25),
            np.linspace(5, 10.8, 30)
        ]),
        'pkt_len': np.random.randint(100, 1500, 55),
        'src_ip': ['192.168.1.2'] * 15 + ['192.168.1.3'] * 10 + ['192.168.1.4'] * 20 + ['192.168.1.5'] * 10,
    }
    conn_df = pd.DataFrame(conn_data)
    return conn_df

def test_slt_extractor_smoke(sample_data):
    """Test that the SLTExtractor runs without raising exceptions."""
    extractor = SLTExtractor(conn_df=sample_data)
    result_df = extractor.process_df()
    assert isinstance(result_df, pd.DataFrame)
    assert not result_df.empty
    assert len(result_df) == sample_data['conn'].nunique()

def test_feature_names_generation():
    """Test the generation of feature names."""
    extractor = SLTExtractor(conn_df=pd.DataFrame())
    feature_names = extractor.feature_names
    assert isinstance(feature_names, list)
    assert len(feature_names) > 1
    assert "conn" in feature_names
    # Check for a representative sample of feature names
    assert 'upstream_ratio_at_2pkt_%' in feature_names
    assert 'upload_timing_4pkt_mean_ms' in feature_names
    assert 'download_throughput_8pkt_bytes_per_sec' in feature_names
    assert 'bidirectional_packet_rate_16pkt_per_sec' in feature_names
    assert 'upload_size_20pkt_std_bytes' in feature_names

def test_process_df(sample_data):
    """Test the main processing loop."""
    extractor = SLTExtractor(conn_df=sample_data)
    result_df = extractor.process_df()
    
    # Ensure all connections are processed
    assert len(result_df) == sample_data['conn'].nunique()
    
    # Check that the output has the correct columns
    assert all(col in result_df.columns for col in extractor.feature_names)

def test_empty_input():
    """Test behavior with an empty DataFrame."""
    conn_df = pd.DataFrame({'conn': [], 'ts_relative': [], 'pkt_len': [], 'src_ip': []})
    extractor = SLTExtractor(conn_df=conn_df)
    result_df = extractor.process_df()
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.empty

def test_connection_with_few_packets(sample_data):
    """Test that connections with fewer packets than checkpoints are handled."""
    # Add a connection with very few packets
    short_conn_data = {
        'conn': ['conn3'] * 3,
        'ts_relative': np.linspace(0, 1, 3),
        'pkt_len': [100, 150, 120],
        'src_ip': ['192.168.1.10'] * 3
    }
    sample_data = pd.concat([sample_data, pd.DataFrame(short_conn_data)], ignore_index=True)
    
    extractor = SLTExtractor(conn_df=sample_data)
    result_df = extractor.process_df()
    
    # All connections, including the short one, should be processed
    assert len(result_df) == sample_data['conn'].nunique()
    
    # Check that the features for the short connection are not all NaN
    short_conn_features = result_df[result_df['conn'] == 'conn3']
    assert not short_conn_features.isnull().values.all() 