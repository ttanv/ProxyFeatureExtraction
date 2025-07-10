"""
Feature Extraction implementation for the Shining Light Into the Tunnel (SLT) paper 
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from feature_extraction.extractors.base_extractor import BaseFeatureExtractor

@dataclass
class PacketFeatures:
    """Data class to store packet features"""
    mean: float
    max: float
    min: float
    std: float

@dataclass
class ConnectionFeatures:
    """Data class to store connection features"""
    upstream_ratio: List[float]
    upload_packet: List[PacketFeatures]
    download_packet: List[PacketFeatures]
    inter_packet: List[PacketFeatures]
    bytes_per_second: Dict[str, List[float]]
    packets_per_second: Dict[str, List[float]]
    size_features: Dict[str, List[PacketFeatures]]

class SLTExtractor(BaseFeatureExtractor):
    """
    Feature extractor class for SLT
    """
    
    def __init__(self, conn_df):
        super().__init__("SLT features", conn_df)
        self.PACKET_CHECKPOINTS = [2, 4, 8, 16, 20]
        self.REQUIRED_LENGTH = 6
        self.MAX_PACKETS = 20
        self.MAX_EMPIRICAL_SAMPLES = 50000
        self.RANDOM_SEED = 42  # Add constant for random seed
        self.feature_names = self._generate_feature_names()
    
    def _generate_feature_names(self):
        """Generate descriptive column names for features"""
        names = ["conn"]
        
        # Upstream ratio features with percentage indicator
        names.extend([f'upstream_ratio_at_{i}pkt_%' for i in self.PACKET_CHECKPOINTS])
        
        # Timing features in milliseconds
        for direction in ['upload', 'download', 'bidirectional']:
            for i in self.PACKET_CHECKPOINTS:
                names.extend([
                    f'{direction}_timing_{i}pkt_mean_ms',
                    f'{direction}_timing_{i}pkt_max_ms',
                    f'{direction}_timing_{i}pkt_min_ms',
                    f'{direction}_timing_{i}pkt_std_ms'
                ])
        
        # Throughput features
        for direction in ['upload', 'download', 'bidirectional']:
            names.extend([f'{direction}_throughput_{i}pkt_bytes_per_sec' 
                         for i in self.PACKET_CHECKPOINTS])
            names.extend([f'{direction}_packet_rate_{i}pkt_per_sec' 
                         for i in self.PACKET_CHECKPOINTS])
        
        # Packet size features in bytes
        for direction in ['upload', 'download', 'bidirectional']:
            for i in self.PACKET_CHECKPOINTS:
                names.extend([
                    f'{direction}_size_{i}pkt_mean_bytes',
                    f'{direction}_size_{i}pkt_max_bytes',
                    f'{direction}_size_{i}pkt_min_bytes',
                    f'{direction}_size_{i}pkt_std_bytes'
                ])
        
        return names
    
    def _calculate_packet_metrics(self, packets: pd.DataFrame) -> Dict:
        """Calculate basic packet metrics"""
        if packets.empty:
            return {'ts': [], 'bytes': [], 'total_bytes': []}
            
        ts = packets['ts_relative'].tolist()
        bytes_data = packets['pkt_len'].tolist()
        
        return {
            'ts': ts,
            'bytes': bytes_data,
            'total_bytes': np.cumsum(bytes_data).tolist()
        }
        
    def _calculate_throughput(self, metrics: Dict) -> Dict[str, List[float]]:
        """Calculate bytes per second for each packet checkpoint"""
        throughput = {
            'upload': [],
            'download': [],
            'inter': []
        }
        
        for direction in throughput.keys():
            direction_metrics = metrics[direction]
            ts = direction_metrics['ts']
            total_bytes = direction_metrics['total_bytes']
            
            for checkpoint in self.PACKET_CHECKPOINTS:
                if checkpoint > len(ts) or checkpoint > len(total_bytes):
                    throughput[direction].append(0)
                else:
                    # Calculate throughput as total_bytes / time_elapsed
                    elapsed_time = ts[checkpoint-1] - ts[0]
                    if elapsed_time > 0:
                        throughput[direction].append(total_bytes[checkpoint-1] / elapsed_time)
                    else:
                        throughput[direction].append(0)
                        
        return throughput
        
    def _calculate_packet_rate(self, metrics: Dict) -> Dict[str, List[float]]:
        """Calculate packets per second for each checkpoint"""
        packet_rates = {
            'upload': [],
            'download': [],
            'inter': []
        }
        
        for direction in packet_rates.keys():
            ts = metrics[direction]['ts']
            
            for checkpoint in self.PACKET_CHECKPOINTS:
                if checkpoint > len(ts):
                    packet_rates[direction].append(0)
                else:
                    # Calculate packet rate as num_packets / time_elapsed
                    elapsed_time = ts[checkpoint-1] - ts[0]
                    if elapsed_time > 0:
                        packet_rates[direction].append(checkpoint / elapsed_time)
                    else:
                        packet_rates[direction].append(0)
                        
        return packet_rates
        
    def _calculate_size_features(self, metrics: Dict) -> Dict[str, List[PacketFeatures]]:
        """Calculate packet size statistics for each checkpoint"""
        size_features = {
            'upload': [],
            'download': [],
            'inter': []
        }
        
        for direction in size_features.keys():
            bytes_data = metrics[direction]['bytes']
            
            for checkpoint in self.PACKET_CHECKPOINTS:
                if checkpoint > len(bytes_data):
                    size_features[direction].append(PacketFeatures(0, 0, 0, 0))
                else:
                    checkpoint_bytes = bytes_data[:checkpoint]
                    size_features[direction].append(PacketFeatures(
                        mean=float(np.mean(checkpoint_bytes)),
                        max=float(np.max(checkpoint_bytes)),
                        min=float(np.min(checkpoint_bytes)),
                        std=float(np.std(checkpoint_bytes))
                    ))
                    
        return size_features
        
    def _calculate_upstream_ratio(self, metrics: Dict) -> List[float]:
        """Calculate upstream ratio at checkpoints"""
        ratios = []
        for checkpoint in self.PACKET_CHECKPOINTS:
            up_bytes = sum(metrics['upload']['bytes'][:checkpoint])
            down_bytes = sum(metrics['download']['bytes'][:checkpoint])
            total = up_bytes + down_bytes
            ratios.append(up_bytes / total if total > 0 else 0)
        return self._pad_list(ratios)
        
    def _calculate_timing_features(self, metrics: Dict) -> List[PacketFeatures]:
        """Calculate timing features at checkpoints"""
        features = []
        ts = metrics['ts']
        
        for checkpoint in self.PACKET_CHECKPOINTS:
            if len(ts) >= checkpoint:
                deltas = np.diff(ts[:checkpoint])
                features.append(PacketFeatures(
                    mean=float(np.mean(deltas)),
                    max=float(np.max(deltas)),
                    min=float(np.min(deltas)),
                    std=float(np.std(deltas))
                ))
            else:
                features.append(PacketFeatures(0, 0, 0, 0))
                
        return features
        
    def _pad_list(self, lst: List, default_value: float = 0) -> List:
        """Pad list to required length"""
        if len(lst) < self.REQUIRED_LENGTH:
            return lst + [default_value] * (self.REQUIRED_LENGTH - len(lst))
        return lst
    
    def _concatenate_features(self, features: ConnectionFeatures) -> List[float]:
        """Flatten ConnectionFeatures into a single list of features.
        
        Args:
            features: ConnectionFeatures object containing all extracted features
            
        Returns:
            List of float values representing all features concatenated in order
        """
        concatenated = []
        
        # Add upstream ratio features
        concatenated.extend(features.upstream_ratio)
        
        # Add timing features for upload, download, and inter-packet
        for packet_features in features.upload_packet:
            concatenated.extend([packet_features.mean, packet_features.max, 
                               packet_features.min, packet_features.std])
        
        for packet_features in features.download_packet:
            concatenated.extend([packet_features.mean, packet_features.max, 
                               packet_features.min, packet_features.std])
        
        for packet_features in features.inter_packet:
            concatenated.extend([packet_features.mean, packet_features.max, 
                               packet_features.min, packet_features.std])
        
        # Add throughput features
        for direction in ['upload', 'download', 'inter']:
            concatenated.extend(features.bytes_per_second[direction])
       
        # Add packet rate features
        for direction in ['upload', 'download', 'inter']:
            concatenated.extend(features.packets_per_second[direction])
        
        # Add size features
        for direction in ['upload', 'download', 'inter']:
            for packet_features in features.size_features[direction]:
                concatenated.extend([packet_features.mean, packet_features.max,
                                   packet_features.min, packet_features.std])
        
        return concatenated
    
    def extract_features(self, conn_data: pd.DataFrame) -> Optional[List[float]]:
        """Extract features from connection data"""
        # set first 6 rows to be the same value 250 for ['pkt_len'] column
        try:
            # Sort and split packets 
            src_ip = conn_data.iloc[0]['src_ip']
            upload_mask = conn_data['src_ip'] == src_ip
            upload_packets = conn_data[upload_mask].head(self.MAX_PACKETS)
            download_packets = conn_data[~upload_mask].head(self.MAX_PACKETS)
            
            # Calculate base metrics
            metrics = {
                'upload': self._calculate_packet_metrics(upload_packets),
                'download': self._calculate_packet_metrics(download_packets),
                'inter': self._calculate_packet_metrics(conn_data.head(self.MAX_PACKETS))
            }
            
            # Calculate features
            features = ConnectionFeatures(
                upstream_ratio=self._calculate_upstream_ratio(metrics),
                upload_packet=self._calculate_timing_features(metrics['upload']),
                download_packet=self._calculate_timing_features(metrics['download']),
                inter_packet=self._calculate_timing_features(metrics['inter']),
                bytes_per_second=self._calculate_throughput(metrics),
                packets_per_second=self._calculate_packet_rate(metrics),
                size_features=self._calculate_size_features(metrics)
            )
            
            return self._concatenate_features(features)
            
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            return None
    
    def process_df(self):
        """
        Goes through each conn in df and applies desired changes
        """
        df_list = []
        for conn, conn_data in self.conn_df.groupby('conn'):
            # Get features for each conn
            conn_features: Optional[List[float]] = self.extract_features(conn_data)
            if not conn_features:
                continue
            
            conn_feature_list = [conn, *conn_features]
            data = {col: [val] for col, val in zip(self.feature_names, conn_feature_list)}
            df_list.append(pd.DataFrame(data))
        
        return pd.concat(df_list) if df_list else pd.DataFrame()