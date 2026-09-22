"""
Feature extraction for network traffic sequences.
Implements the 11 statistical features described in the paper.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class TrafficFeatureExtractor:
    """
    Extracts 11 statistical features from network traffic flows:
    1. Mean packet length
    2. Variance of packet lengths
    3. Mean inter-arrival time
    4. Variance of inter-arrival times
    5. Total bytes
    6. Total packets
    7. Cumulative flow duration
    8. Percentage of TCP SYN flags
    9. Percentage of ACK flags
    10. Entropy of packet sizes
    11. Burstiness coefficient (peak-to-mean arrival rate ratio)
    """
    
    FEATURE_NAMES = [
        "mean_packet_length",
        "var_packet_length",
        "mean_iat",
        "var_iat",
        "total_bytes",
        "total_packets",
        "flow_duration",
        "syn_flag_ratio",
        "ack_flag_ratio",
        "packet_size_entropy",
        "burstiness",
    ]
    
    def __init__(self, window_size: float = 1.0):
        """
        Args:
            window_size: temporal aggregation window in seconds
        """
        self.window_size = window_size
        self.num_features = len(self.FEATURE_NAMES)
        
    def extract_from_flow(self, flow_data: pd.DataFrame) -> np.ndarray:
        """
        Extract features from a single flow.
        
        Args:
            flow_data: DataFrame with columns [timestamp, packet_length, direction, flags]
        Returns:
            features: (11,) array
        """
        if len(flow_data) == 0:
            return np.zeros(self.num_features)
        
        packet_lengths = flow_data["packet_length"].values
        timestamps = flow_data["timestamp"].values
        
        # Inter-arrival times
        if len(timestamps) > 1:
            iat = np.diff(timestamps)
        else:
            iat = np.array([0.0])
        
        # Basic statistics
        mean_pkt_len = np.mean(packet_lengths)
        var_pkt_len = np.var(packet_lengths)
        mean_iat = np.mean(iat)
        var_iat = np.var(iat)
        total_bytes = np.sum(packet_lengths)
        total_packets = len(packet_lengths)
        flow_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
        
        # Flag ratios
        if "flags" in flow_data.columns:
            flags = flow_data["flags"].values
            syn_count = np.sum(flags == "SYN")
            ack_count = np.sum(flags == "ACK")
            syn_ratio = syn_count / max(total_packets, 1)
            ack_ratio = ack_count / max(total_packets, 1)
        else:
            syn_ratio = 0.0
            ack_ratio = 0.0
        
        # Packet size entropy
        if total_packets > 0:
            hist, _ = np.histogram(packet_lengths, bins=10, density=True)
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
        else:
            entropy = 0.0
        
        # Burstiness coefficient
        if flow_duration > 0 and total_packets > 1:
            arrival_rate = total_packets / flow_duration
            # Peak rate: max packets in any 1-second window
            peak_rate = self._compute_peak_rate(timestamps)
            burstiness = peak_rate / max(arrival_rate, 1e-10)
        else:
            burstiness = 0.0
        
        features = np.array([
            mean_pkt_len,
            var_pkt_len,
            mean_iat,
            var_iat,
            total_bytes,
            total_packets,
            flow_duration,
            syn_ratio,
            ack_ratio,
            entropy,
            burstiness,
        ], dtype=np.float32)
        
        return features
    
    def _compute_peak_rate(self, timestamps: np.ndarray) -> float:
        """Compute peak arrival rate in packets/second."""
        if len(timestamps) < 2:
            return 0.0
        
        # Count packets in sliding 1-second windows
        max_count = 0
        for i in range(len(timestamps)):
            count = np.sum(
                (timestamps >= timestamps[i]) &
                (timestamps < timestamps[i] + self.window_size)
            )
            max_count = max(max_count, count)
        
        return max_count / self.window_size
    
    def extract_from_dataframe(
        self,
        df: pd.DataFrame,
        flow_id_col: str = "flow_id",
        timestamp_col: str = "timestamp",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from all flows in a DataFrame.
        
        Returns:
            features: (num_flows, 11)
            flow_ids: (num_flows,)
        """
        features_list = []
        flow_ids = []
        
        for flow_id, group in df.groupby(flow_id_col):
            group = group.sort_values(timestamp_col)
            feat = self.extract_from_flow(group)
            features_list.append(feat)
            flow_ids.append(flow_id)
        
        return np.array(features_list), np.array(flow_ids)
