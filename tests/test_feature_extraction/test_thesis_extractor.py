import logging
import pytest
import pandas as pd
import numpy as np

from feature_extraction.extractors.thesis_extractor import ThesisExtractor

def sequential_rtt_ratio_analyzer(df, pkt_limit):
    """
    A sequential, pandas/numpy-based implementation of the RTT ratio logic
    for verification purposes.
    """
    all_ratios = {}

    for conn_id, group_df in df.groupby('conn'):
        packets = group_df.to_dict('records')

        if len(packets) < pkt_limit:
            continue
        
        group_df = group_df.head(pkt_limit)
        packets = group_df.to_dict('records')


        try:
            original_src_ip = packets[0]['src_ip']
            original_dst_ip = packets[0]['dst_ip']
        except (IndexError, KeyError):
            continue

        found_triplets = []
        for i in range(len(packets) - 2):
            p1, p2, p3 = packets[i], packets[i+1], packets[i+2]
            original_src_ip = p1['dst_ip']
            original_dst_ip = p1['src_ip']

            is_p1_correct = (p1['src_ip'] == original_dst_ip and p1['dst_ip'] == original_src_ip)
            is_p2_correct = (p2['src_ip'] == original_src_ip and p2['dst_ip'] == original_dst_ip)
            is_p3_correct = (p3['src_ip'] == original_src_ip and p3['dst_ip'] == original_dst_ip)

            if is_p1_correct and is_p2_correct and is_p3_correct:
                found_triplets.append((p1, p2, p3))

        if len(found_triplets) >= 2:
            first_triplet = found_triplets[0]
            second_triplet = found_triplets[1]

            try:
                ts1_p1 = float(first_triplet[0]['ts_relative'])
                ts1_p2 = float(first_triplet[1]['ts_relative'])
                
                ts2_p1 = float(second_triplet[0]['ts_relative'])
                ts2_p3 = float(second_triplet[2]['ts_relative'])

                delta1 = ts1_p2 - ts1_p1
                delta2 = ts2_p3 - ts2_p1
                
                if delta2 != 0:
                    ratio = delta1 / delta2
                    all_ratios[conn_id] = ratio

            except (ValueError, KeyError):
                continue
    
    if all_ratios:
        return pd.DataFrame([{"conn": name, "rtt_ratio": value} for name, value in all_ratios.items()])
    else:
        return pd.DataFrame()


@pytest.fixture
def sample_data():
    """Pytest fixture to create a sample DataFrame for testing."""
    
    # conn1: Perfect case with 2 triplets
    conn1_packets = [
        # 1st triplet
        {'conn': 'conn1', 'src_ip': 'B', 'dst_ip': 'A', 'ts_relative': 1.0},
        {'conn': 'conn1', 'src_ip': 'A', 'dst_ip': 'B', 'ts_relative': 1.1}, # delta1 = 0.1
        {'conn': 'conn1', 'src_ip': 'A', 'dst_ip': 'B', 'ts_relative': 1.2},
        # Some other packets
        {'conn': 'conn1', 'src_ip': 'A', 'dst_ip': 'B', 'ts_relative': 1.3},
        # 2nd triplet
        {'conn': 'conn1', 'src_ip': 'B', 'dst_ip': 'A', 'ts_relative': 2.0},
        {'conn': 'conn1', 'src_ip': 'A', 'dst_ip': 'B', 'ts_relative': 2.2},
        {'conn': 'conn1', 'src_ip': 'A', 'dst_ip': 'B', 'ts_relative': 2.4}, # delta2 = 0.4
        # ratio = 0.1 / 0.4 = 0.25
    ]
    
    # conn2: Only one triplet
    conn2_packets = [
        {'conn': 'conn2', 'src_ip': 'C', 'dst_ip': 'D', 'ts_relative': 1.0},
        {'conn': 'conn2', 'src_ip': 'D', 'dst_ip': 'C', 'ts_relative': 1.1},
        {'conn': 'conn2', 'src_ip': 'D', 'dst_ip': 'C', 'ts_relative': 1.2},
        {'conn': 'conn2', 'src_ip': 'D', 'dst_ip': 'C', 'ts_relative': 1.3},
    ]

    # conn3: Not enough packets (< 20 for default limit)
    conn3_packets = [{'conn': 'conn3', 'src_ip': 'E', 'dst_ip': 'F', 'ts_relative': i} for i in range(10)]

    # conn4: Zero delta2
    conn4_packets = [
        # 1st triplet
        {'conn': 'conn4', 'src_ip': 'B', 'dst_ip': 'A', 'ts_relative': 1.0},
        {'conn': 'conn4', 'src_ip': 'A', 'dst_ip': 'B', 'ts_relative': 1.1},
        {'conn': 'conn4', 'src_ip': 'A', 'dst_ip': 'B', 'ts_relative': 1.2},
        # 2nd triplet with zero delta
        {'conn': 'conn4', 'src_ip': 'B', 'dst_ip': 'A', 'ts_relative': 2.0},
        {'conn': 'conn4', 'src_ip': 'A', 'dst_ip': 'B', 'ts_relative': 2.2},
        {'conn': 'conn4', 'src_ip': 'A', 'dst_ip': 'B', 'ts_relative': 2.0}, # Same timestamp
    ]
    
    # Make sure we have enough packets to pass pkt_limit
    conn1_packets.extend([{'conn': 'conn1', 'src_ip': 'X', 'dst_ip': 'Y', 'ts_relative': 10 + i} for i in range(20)])
    conn2_packets.extend([{'conn': 'conn2', 'src_ip': 'X', 'dst_ip': 'Y', 'ts_relative': 10 + i} for i in range(20)])
    conn4_packets.extend([{'conn': 'conn4', 'src_ip': 'X', 'dst_ip': 'Y', 'ts_relative': 10 + i} for i in range(20)])


    all_packets = conn1_packets + conn2_packets + conn3_packets + conn4_packets
    return pd.DataFrame(all_packets)

def test_smoke_and_no_errors(sample_data):
    """
    Test that the feature extractor runs without raising any exceptions.
    """
    extractor = ThesisExtractor(conn_df=sample_data)
    result_df = extractor.process_df()

    assert isinstance(result_df, pd.DataFrame)
    assert not result_df.empty
    logging.debug(result_df['conn'])
    # assert 'conn1' in result_df['conn']
    assert result_df['conn'].isin(['conn1']).any()
    assert not result_df['conn'].isin(['conn2']).any()
    assert not result_df['conn'].isin(['conn3']).any()
    assert not result_df['conn'].isin(['conn4']).any()

def test_correctness_against_sequential(sample_data):
    """
    Test that the implementation matches the sequential implementation.
    """
    pkt_limit = 20

    # Get result from the class implementation
    extractor = ThesisExtractor(conn_df=sample_data)
    result_df = extractor.process_df(pkt_limit=pkt_limit)

    # Get result from the sequential CPU implementation
    sequential_result = sequential_rtt_ratio_analyzer(sample_data, pkt_limit=pkt_limit)
    
    # print(result_df.sort_index(axis=1))
    print(sequential_result['rtt_ratio'].dtype)
    
    
    pd.testing.assert_frame_equal(result_df.sort_index(axis=1), sequential_result.sort_index(axis=1), check_dtype=True, atol=1e-5)
    
def test_empty_input():
    """
    Test the behavior with empty input DataFrames.
    """
    conn_df = pd.DataFrame({'conn': [], 'src_ip': [], 'dst_ip': [], 'ts_relative': []})
    
    extractor = ThesisExtractor(conn_df=conn_df)
    result_df = extractor.process_df(pkt_limit=20)

    assert isinstance(result_df, pd.DataFrame)
    assert result_df.empty

def test_insufficient_packets(sample_data):
    """
    Test that connections with fewer packets than pkt_limit are ignored.
    """
    extractor = ThesisExtractor(conn_df=sample_data)
    result_df = extractor.process_df(pkt_limit=20) # conn3 has 10 packets
    assert 'conn3' not in result_df.columns

def test_insufficient_triplets(sample_data):
    """
    Test connections with fewer than two 'perfect round trip' triplets.
    """
    extractor = ThesisExtractor(conn_df=sample_data)
    result_df = extractor.process_df(pkt_limit=20) # conn2 has one triplet
    assert 'conn2' not in result_df.columns
    
def test_zero_delta2(sample_data):
    """
    Test the case where the second delta is zero, preventing division by zero.
    """
    extractor = ThesisExtractor(conn_df=sample_data)
    result_df = extractor.process_df(pkt_limit=20) # conn4 has zero delta2
    assert 'conn4' not in result_df.columns

def test_output_format_and_values(sample_data):
    """
    Test the output format and the correctness of the calculated ratio.
    """
    extractor = ThesisExtractor(conn_df=sample_data)
    result_df = extractor.process_df(pkt_limit=20)
    
    # Expected ratio for conn1: delta1 = 0.1, delta2 = 0.4, ratio = 0.25
    expected_ratio = 0.25
    assert result_df['conn'].isin(['conn1']).any()
    assert np.isclose(result_df[result_df['conn'] == 'conn1']['rtt_ratio'], expected_ratio)
    assert len(result_df.columns) == 2
    assert result_df.shape[0] == 1 