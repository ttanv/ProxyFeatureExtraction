from abc import ABC, abstractmethod
import logging

class BaseFeatureExtractor(ABC):
    """Abstract base class for all feature extractors"""
    
    def __init__(self, feature_name, conn_df, gateway_df=None):
        self.feature_name = feature_name
        self.conn_df = conn_df
        self.gateway_df = gateway_df
    
    @abstractmethod
    def process_df(pkt_limit=20):
        """
        Abstract method: takes a pcap_df and returns a df with a single row
        for each connection along with a set of extracted features
        """
        raise NotImplementedError("This method should be overridden by subclasses")
    
    
    
    