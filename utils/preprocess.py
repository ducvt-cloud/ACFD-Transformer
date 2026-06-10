import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

class StreamPreprocessor:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.scaler = MinMaxScaler()

    def process_csv(self, file_path):
        """Transforms raw flow records into sliding window sequences."""
        df = pd.read_csv(file_path)
        # Drop non-feature columns
        X_raw = df.iloc[:, :-2].values 
        y_raw = df.iloc[:, -1].values
        
        X_scaled = self.scaler.fit_transform(X_raw)
        
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - self.window_size):
            X_seq.append(X_scaled[i : i + self.window_size])
            y_seq.append(y_raw[i + self.window_size])
            
        return torch.FloatTensor(np.array(X_seq)), torch.LongTensor(np.array(y_seq))
