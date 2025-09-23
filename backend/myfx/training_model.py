
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from .models import ForexData
from .utils import fetch_xauusd

MODEL_PATH = "xauusd_nn.pt"

class ForexNN(nn.Module):
    def __init__(self):
        super(ForexNN, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

def load_data(fetch_if_empty=True, normalize=True):
    """
    Load XAU/USD data from DB.
    - fetch_if_empty: automatically fetch data if table is empty
    - normalize: scale close prices to 0-1 for training
    """
    if ForexData.objects.count() == 0 and fetch_if_empty:
        fetch_xauusd()

    qs = ForexData.objects.all().order_by("date")
    df = pd.DataFrame(list(qs.values("date", "close")))
    if df.empty:
        return None, None, None, None, None

    df["date"] = pd.to_datetime(df["date"])
    
    # Use day index for input
    X = torch.arange(len(df), dtype=torch.float32).unsqueeze(1)
    
    # Normalize close prices
    y = torch.tensor(df["close"].values, dtype=torch.float32).unsqueeze(1)
    if normalize:
        y_min, y_max = y.min(), y.max()
        y_norm = (y - y_min) / (y_max - y_min)
    else:
        y_norm = y
        y_min, y_max = None, None

    return X, y_norm, df, y_min, y_max

def train_nn(epochs=200, lr=0.001):
    X, y, df, _, _ = load_data()
    if X is None:
        print("No data to train")
        return None

    model = ForexNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X)
            val_loss = criterion(val_outputs, y)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Train Loss: {loss.item():.4f}, Eval Loss: {val_loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Neuro Network saved to {MODEL_PATH}")

    return model

def predict_next(days=5, normalize=True):
    X, y, df, y_min, y_max = load_data(fetch_if_empty=True, normalize=normalize)
    if X is None:
        return []

    model = ForexNN()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    last_index = X[-1].item()
    last_date = df["date"].iloc[-1]

    preds = []
    for i in range(1, days + 1):
        future_index = torch.tensor([[last_index + i]], dtype=torch.float32)
        pred = model(future_index)
        if normalize and y_min is not None and y_max is not None:
            pred = pred * (y_max - y_min) + y_min  # denormalize
        preds.append({
            "date": str(last_date + pd.Timedelta(days=i)),
            "predicted_close": round(pred.item(), 2)  # convert tensor to float
        })

    return preds

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# import numpy as np
# import os
# import pickle
# from .models import ForexData
# from .utils import fetch_xauusd

# MODEL_PATH = "xauusd_lstm.pt"
# SCALER_PATH = "xauusd_scaler.pkl"
# SEQUENCE_LENGTH = 15  # number of days to look back

# class ForexNN(nn.Module):
#     def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.2):
#         super(ForexNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
#                             batch_first=True, dropout=dropout if num_layers>1 else 0)
#         self.fc_layers = nn.Sequential(
#             nn.Linear(hidden_size, 32),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(16, 1)
#         )

#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         lstm_out, _ = self.lstm(x, (h0, c0))
#         last_output = lstm_out[:, -1, :]
#         return self.fc_layers(last_output)

# def create_features(df):
#     """Add technical indicators and prepare features"""
#     df = df.copy()
#     for col in ["open", "high", "low", "close"]:
#         if col not in df.columns:
#             df[col] = df["close"].shift(1).fillna(df["close"])
#     df["ma_5"] = df["close"].rolling(5, min_periods=1).mean()
#     df = df.fillna(method='ffill').fillna(method='bfill')
#     return df

# class SimpleScaler:
#     def __init__(self):
#         self.data_min_ = None
#         self.data_max_ = None
#         self.data_range_ = None
        
#     def fit(self, X):
#         self.data_min_ = X.min(axis=0)
#         self.data_max_ = X.max(axis=0)
#         self.data_range_ = self.data_max_ - self.data_min_
#         self.data_range_[self.data_range_==0] = 1.0
#         return self
    
#     def transform(self, X):
#         if self.data_min_ is None or self.data_range_ is None:
#             raise ValueError("Scaler must be fitted before transforming")
#         return (X - self.data_min_) / self.data_range_
    
#     def fit_transform(self, X):
#         return self.fit(X).transform(X)
    
#     def inverse_transform(self, X):
#         if self.data_min_ is None or self.data_range_ is None:
#             raise ValueError("Scaler must be fitted before inverse transforming")
#         return X * self.data_range_ + self.data_min_

# def create_sequences(data, sequence_length=SEQUENCE_LENGTH):
#     X, y = [], []
#     for i in range(sequence_length, len(data)):
#         X.append(data[i-sequence_length:i])
#         y.append(data[i, 0])  # predict close price
#     return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# def load_data(sequence_length=SEQUENCE_LENGTH, test_split=0.2, fetch_if_empty=True):
#     if ForexData.objects.count() == 0 and fetch_if_empty:
#         fetch_xauusd()
    
#     qs = ForexData.objects.all().order_by("date")
#     df = pd.DataFrame(list(qs.values("date", "open", "high", "low", "close")))
    
#     if df.empty or len(df) < sequence_length + 1:
#         print("Not enough data")
#         return None, None, None, None, None
    
#     df["date"] = pd.to_datetime(df["date"])
#     df = create_features(df)
    
#     features = df[["close", "open", "high", "low", "ma_5"]].values.astype(np.float32)
#     scaler = SimpleScaler()
#     features_scaled = scaler.fit_transform(features)
    
#     X, y = create_sequences(features_scaled, sequence_length)
    
#     split_idx = int(len(X) * (1 - test_split))
#     X_train, X_test = X[:split_idx], X[split_idx:]
#     y_train, y_test = y[:split_idx], y[split_idx:]
    
#     return (torch.FloatTensor(X_train), torch.FloatTensor(X_test),
#             torch.FloatTensor(y_train).unsqueeze(1), torch.FloatTensor(y_test).unsqueeze(1),
#             scaler, df)

# def save_model_and_scaler(model, scaler, input_size=5):
#     """Save model and scaler separately following best practices"""
#     # Save only PyTorch-native objects in model checkpoint
#     model_checkpoint = {
#         'model_state_dict': model.state_dict(),
#         'model_config': {
#             'input_size': input_size,
#             'hidden_size': model.hidden_size,
#             'num_layers': model.num_layers,
#         },
#         'feature_columns': ["close", "open", "high", "low", "ma_5"],
#         'sequence_length': SEQUENCE_LENGTH,
#         'pytorch_version': str(torch.__version__)  # Convert to string to avoid TorchVersion object
#     }
    
#     # Save model with weights_only=True compatibility
#     torch.save(model_checkpoint, MODEL_PATH)
    
#     # Save scaler separately using pickle
#     with open(SCALER_PATH, 'wb') as f:
#         pickle.dump(scaler, f)
    
#     print(f"Model saved to {MODEL_PATH}")
#     print(f"Scaler saved to {SCALER_PATH}")

# def load_model_and_scaler():
#     """Load model and scaler separately"""
#     if not os.path.exists(MODEL_PATH):
#         raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
#     if not os.path.exists(SCALER_PATH):
#         raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
    
#     # Add safe globals for any legacy TorchVersion objects in old checkpoints
#     torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    
#     # Load model checkpoint (weights_only=True by default in PyTorch 2.6+)
#     checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
#     # Load scaler separately
#     with open(SCALER_PATH, 'rb') as f:
#         scaler = pickle.load(f)
    
#     # Create model from saved config
#     model_config = checkpoint['model_config']
#     model = ForexNN(
#         input_size=model_config['input_size'],
#         hidden_size=model_config['hidden_size'],
#         num_layers=model_config['num_layers']
#     )
    
#     # Load model weights
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
    
#     return model, scaler, checkpoint

# def train_nn(epochs=200, lr=0.001):
#     """Train neural network with proper model saving"""
#     X_train, X_test, y_train, y_test, scaler, df = load_data()
#     if X_train is None:
#         return None

#     model = ForexNN(input_size=5)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
#     best_val_loss = float('inf')
#     patience_counter = 0
#     patience = 30
    
#     for epoch in range(epochs):
#         # Training phase
#         model.train()
#         optimizer.zero_grad()
#         outputs = model(X_train)
#         loss = criterion(outputs, y_train)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
        
#         # Validation phase
#         model.eval()
#         with torch.no_grad():
#             val_outputs = model(X_test)
#             val_loss = criterion(val_outputs, y_test)
        
#         scheduler.step(val_loss)
        
#         # Save best model
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             patience_counter = 0
#             # Use the new saving function
#             save_model_and_scaler(model, scaler, input_size=5)
#         else:
#             patience_counter += 1
        
#         if epoch % 10 == 0:
#             print(f"Epoch {epoch}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
#     print(f"The best valid loss: {best_val_loss}")
        
#         # if patience_counter >= patience:
#         #     print(f"Early stopping at epoch {epoch}")
#         #     break
    
#     return model

# def predict_next(days=5, sequence_length=15):
#     """Predict future prices with proper error handling"""
#     try:
#         # Load model and scaler using the new function
#         model, scaler, checkpoint = load_model_and_scaler()
        
#         # Get configuration from checkpoint
#         feature_columns = checkpoint['feature_columns']
#         saved_sequence_length = checkpoint.get('sequence_length', sequence_length)
        
#         # Load data from database
#         qs = ForexData.objects.all().order_by("date")
#         field_names = [f.name for f in ForexData._meta.get_fields()]
#         values_list = ["date", "close"]
#         optional_fields = ["open", "high", "low", "volume"]
#         for field in optional_fields:
#             if field in field_names:
#                 values_list.append(field)
        
#         df = pd.DataFrame(list(qs.values(*values_list)))
        
#         if df.empty or len(df) < saved_sequence_length:
#             print(f"Not enough data for prediction. Need at least {saved_sequence_length} records.")
#             return []
        
#         # Prepare data
#         df["date"] = pd.to_datetime(df["date"])
#         df = df.sort_values("date").reset_index(drop=True)
#         df = create_features(df)
        
#         # Transform features
#         features = df[feature_columns].values.astype(np.float32)
#         features_scaled = scaler.transform(features)
#         last_sequence = features_scaled[-saved_sequence_length:]
        
#         # Make predictions
#         predictions = []
#         current_sequence = last_sequence.copy()
#         last_date = df["date"].iloc[-1]
        
#         with torch.no_grad():
#             for i in range(days):
#                 # Predict next value
#                 input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0)
#                 pred_scaled = model(input_tensor).item()
                
#                 # Inverse transform prediction
#                 dummy_features = np.zeros((1, len(feature_columns)))
#                 dummy_features[0, 0] = pred_scaled
#                 pred_price = scaler.inverse_transform(dummy_features)[0, 0]
                
#                 predictions.append({
#                     "date": str(last_date + pd.Timedelta(days=i+1)),
#                     "predicted_close": round(pred_price, 2)
#                 })
                
#                 # Update sequence for next prediction
#                 next_features = current_sequence[-1].copy()
#                 next_features[0] = pred_scaled
#                 if len(feature_columns) > 1:
#                     next_features[1] = (next_features[1] * 4 + pred_scaled) / 5  # ma_5 approximation
                
#                 current_sequence = np.roll(current_sequence, -1, axis=0)
#                 current_sequence[-1] = next_features
        
#         return predictions
        
#     except FileNotFoundError as e:
#         print(f"Model files not found: {e}")
#         print("Please train the model first using train_nn()")
#         return []
#     except Exception as e:
#         print(f"Error in prediction: {e}")
#         import traceback
#         traceback.print_exc()
#         return []

# def get_model_info():
#     """Get information about the saved model"""
#     try:
#         if not os.path.exists(MODEL_PATH):
#             return "No model found. Train a model first."
        
#         checkpoint = torch.load(MODEL_PATH, map_location='cpu')
#         return {
#             "model_config": checkpoint.get('model_config', {}),
#             "feature_columns": checkpoint.get('feature_columns', []),
#             "sequence_length": checkpoint.get('sequence_length', 'Unknown'),
#             "pytorch_version": checkpoint.get('pytorch_version', 'Unknown'),
#             "file_size_mb": round(os.path.getsize(MODEL_PATH) / (1024*1024), 2)
#         }
#     except Exception as e:
#         return f"Error reading model info: {e}"
"""
version 3 - get current data
"""
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# import numpy as np
# import os
# import pickle
# import requests
# from django.utils.dateparse import parse_datetime
# from datetime import datetime
# import logging
# from .models import ForexData

# MODEL_PATH = "xauusd_lstm.pt"
# SCALER_PATH = "xauusd_scaler.pkl"
# SEQUENCE_LENGTH = 15  # Number of days to look back

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # --- Neural Network Class (Unchanged) ---
# class ForexNN(nn.Module):
#     def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.2):
#         super(ForexNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
#                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
#         self.fc_layers = nn.Sequential(
#             nn.Linear(hidden_size, 32),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(16, 1)
#         )

#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         lstm_out, _ = self.lstm(x, (h0, c0))
#         last_output = lstm_out[:, -1, :]
#         return self.fc_layers(last_output)

# # --- Feature Creation (Unchanged) ---
# def create_features(df):
#     """Add technical indicators and prepare features"""
#     df = df.copy()
#     for col in ["open", "high", "low"]:
#         if col not in df.columns:
#             df[col] = df["close"].shift(1).fillna(df["close"])
#     df["ma_5"] = df["close"].rolling(5, min_periods=1).mean()
#     df = df.fillna(method='ffill').fillna(method='bfill')
#     return df

# # --- Scaler Class (Unchanged) ---
# class SimpleScaler:
#     def __init__(self):
#         self.data_min_ = None
#         self.data_max_ = None
#         self.data_range_ = None
        
#     def fit(self, X):
#         self.data_min_ = X.min(axis=0)
#         self.data_max_ = X.max(axis=0)
#         self.data_range_ = self.data_max_ - self.data_min_
#         self.data_range_[self.data_range_ == 0] = 1.0
#         return self
    
#     def transform(self, X):
#         if self.data_min_ is None or self.data_range_ is None:
#             raise ValueError("Scaler must be fitted before transforming")
#         return (X - self.data_min_) / self.data_range_
    
#     def fit_transform(self, X):
#         return self.fit(X).transform(X)
    
#     def inverse_transform(self, X):
#         if self.data_min_ is None or self.data_range_ is None:
#             raise ValueError("Scaler must be fitted before inverse transforming")
#         return X * self.data_range_ + self.data_min_

# # --- Sequence Creation (Unchanged) ---
# def create_sequences(data, sequence_length=SEQUENCE_LENGTH):
#     X, y = [], []
#     for i in range(sequence_length, len(data)):
#         X.append(data[i-sequence_length:i])
#         y.append(data[i, 0])  # Predict close price
#     return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# # --- Updated fetch_xauusd (As Defined Above) ---
# def fetch_xauusd(latest_only=True, min_data_points=SEQUENCE_LENGTH + 1):
#     """
#     Fetch XAU/USD daily data from Twelve Data.
#     Args:
#         latest_only (bool): If True, fetch only data newer than the latest in DB.
#         min_data_points (int): Minimum number of data points required.
#     Returns:
#         bool: True if data was successfully fetched and saved, False otherwise.
#     """
#     try:
#         url = 'https://api.twelvedata.com/time_series'
#         params = {
#             "symbol": "XAU/USD",
#             "interval": "1day",
#             "outputsize": 30 if latest_only else max(30, min_data_points * 2),
#             "apikey": "92316069f35c407aa1ebdbd7279fe144",
#         }

#         response = requests.get(url, params=params, timeout=10)
#         response.raise_for_status()
#         data = response.json()

#         if 'values' not in data or not data['values']:
#             logger.error(f"Error fetching XAU/USD: {data.get('message', 'No values returned')}")
#             return False

#         latest_in_db = ForexData.objects.order_by('-date').first()
#         latest_date = latest_in_db.date if latest_in_db else None

#         saved_count = 0
#         for entry in data['values']:
#             dt = parse_datetime(entry["datetime"])
#             if dt is None:
#                 logger.warning(f"Invalid datetime format: {entry['datetime']}")
#                 continue

#             dt = dt.replace(tzinfo=None) if dt.tzinfo else dt

#             if latest_only and latest_date and dt.date() <= latest_date:
#                 continue

#             ForexData.objects.update_or_create(
#                 date=dt.date(),
#                 defaults={
#                     "open": float(entry["open"]),
#                     "high": float(entry["high"]),
#                     "low": float(entry["low"]),
#                     "close": float(entry["close"]),
#                     "volume": float(entry.get("volume", 0))
#                 }
#             )
#             saved_count += 1

#         logger.info(f"XAU/USD: Fetched and saved {saved_count} new records.")
#         return saved_count > 0

#     except requests.exceptions.RequestException as e:
#         logger.error(f"Network error fetching XAU/USD: {e}")
#         return False
#     except Exception as e:
#         logger.error(f"Unexpected error fetching XAU/USD: {e}")
#         return False

# # --- Updated load_data Function ---
# def load_data(sequence_length=SEQUENCE_LENGTH, test_split=0.2, fetch_if_empty=True):
#     """
#     Load data from ForexData model, fetching new data if necessary.
#     """
#     # Fetch data if the database is empty or has insufficient records
#     if fetch_if_empty or ForexData.objects.count() < sequence_length + 1:
#         success = fetch_xauusd(latest_only=True, min_data_points=sequence_length + 1)
#         if not success:
#             logger.error("Failed to fetch current XAU/USD data.")
#             return None, None, None, None, None
    
#     qs = ForexData.objects.all().order_by("date")
#     df = pd.DataFrame(list(qs.values("date", "open", "high", "low", "close", "volume")))
    
#     if df.empty or len(df) < sequence_length + 1:
#         logger.error(f"Not enough data: {len(df)} records. Need at least {sequence_length + 1}.")
#         return None, None, None, None, None
    
#     df["date"] = pd.to_datetime(df["date"])
#     df = create_features(df)
    
#     features = df[["close", "open", "high", "low", "ma_5"]].values.astype(np.float32)
#     scaler = SimpleScaler()
#     features_scaled = scaler.fit_transform(features)
    
#     X, y = create_sequences(features_scaled, sequence_length)
    
#     split_idx = int(len(X) * (1 - test_split))
#     X_train, X_test = X[:split_idx], X[split_idx:]
#     y_train, y_test = y[:split_idx], y[split_idx:]
    
#     return (torch.FloatTensor(X_train), torch.FloatTensor(X_test),
#             torch.FloatTensor(y_train).unsqueeze(1), torch.FloatTensor(y_test).unsqueeze(1),
#             scaler, df)

# # --- Save Model and Scaler (Unchanged) ---
# def save_model_and_scaler(model, scaler, input_size=5):
#     """Save model and scaler separately following best practices"""
#     model_checkpoint = {
#         'model_state_dict': model.state_dict(),
#         'model_config': {
#             'input_size': input_size,
#             'hidden_size': model.hidden_size,
#             'num_layers': model.num_layers,
#         },
#         'feature_columns': ["close", "open", "high", "low", "ma_5"],
#         'sequence_length': SEQUENCE_LENGTH,
#         'pytorch_version': str(torch.__version__)
#     }
    
#     torch.save(model_checkpoint, MODEL_PATH)
    
#     with open(SCALER_PATH, 'wb') as f:
#         pickle.dump(scaler, f)
    
#     logger.info(f"Model saved to {MODEL_PATH}")
#     logger.info(f"Scaler saved to {SCALER_PATH}")

# # --- Load Model and Scaler (Unchanged) ---
# def load_model_and_scaler():
#     """Load model and scaler separately"""
#     if not os.path.exists(MODEL_PATH):
#         raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
#     if not os.path.exists(SCALER_PATH):
#         raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
    
#     torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    
#     checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
#     with open(SCALER_PATH, 'rb') as f:
#         scaler = pickle.load(f)
    
#     model_config = checkpoint['model_config']
#     model = ForexNN(
#         input_size=model_config['input_size'],
#         hidden_size=model_config['hidden_size'],
#         num_layers=model_config['num_layers']
#     )
    
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
    
#     return model, scaler, checkpoint

# # --- Train Neural Network (Unchanged) ---
# def train_nn(epochs=200, lr=0.001):
#     """Train neural network with proper model saving"""
#     X_train, X_test, y_train, y_test, scaler, df = load_data()
#     if X_train is None:
#         logger.error("Training aborted: No data available.")
#         return None

#     model = ForexNN(input_size=5)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
#     best_val_loss = float('inf')
#     patience_counter = 0
#     patience = 30
    
#     for epoch in range(epochs):
#         model.train()
#         optimizer.zero_grad()
#         outputs = model(X_train)
#         loss = criterion(outputs, y_train)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
        
#         model.eval()
#         with torch.no_grad():
#             val_outputs = model(X_test)
#             val_loss = criterion(val_outputs, y_test)
        
#         scheduler.step(val_loss)
        
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             patience_counter = 0
#             save_model_and_scaler(model, scaler, input_size=5)
#         else:
#             patience_counter += 1
        
#         if epoch % 10 == 0:
#             logger.info(f"Epoch {epoch}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
    
#     logger.info(f"The best valid loss: {best_val_loss}")
#     return model

# # --- Updated predict_next Function ---
# def predict_next(days=5, sequence_length=SEQUENCE_LENGTH):
#     """Predict future prices with current data"""
#     try:
#         # Ensure fresh data
#         success = fetch_xauusd(latest_only=True, min_data_points=sequence_length + 1)
#         if not success:
#             logger.error("Failed to fetch current data for prediction.")
#             return []
        
#         # Load model and scaler
#         model, scaler, checkpoint = load_model_and_scaler()
        
#         feature_columns = checkpoint['feature_columns']
#         saved_sequence_length = checkpoint.get('sequence_length', sequence_length)
        
#         # Load data from database
#         qs = ForexData.objects.all().order_by("date")
#         field_names = [f.name for f in ForexData._meta.get_fields()]
#         values_list = ["date", "close"]
#         optional_fields = ["open", "high", "low", "volume"]
#         for field in optional_fields:
#             if field in field_names:
#                 values_list.append(field)
        
#         df = pd.DataFrame(list(qs.values(*values_list)))
        
#         if df.empty or len(df) < saved_sequence_length:
#             logger.error(f"Not enough data for prediction. Need at least {saved_sequence_length} records, got {len(df)}.")
#             return []
        
#         df["date"] = pd.to_datetime(df["date"])
#         df = df.sort_values("date").reset_index(drop=True)
#         df = create_features(df)
        
#         # Transform features
#         features = df[feature_columns].values.astype(np.float32)
#         features_scaled = scaler.transform(features)
#         last_sequence = features_scaled[-saved_sequence_length:]
        
#         # Make predictions
#         predictions = []
#         current_sequence = last_sequence.copy()
#         last_date = df["date"].iloc[-1]
        
#         with torch.no_grad():
#             for i in range(days):
#                 # Predict next value
#                 input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0)
#                 pred_scaled = model(input_tensor).item()
                
#                 # Inverse transform prediction
#                 dummy_features = np.zeros((1, len(feature_columns)))
#                 dummy_features[0, 0] = pred_scaled
#                 pred_price = scaler.inverse_transform(dummy_features)[0, 0]
                
#                 predictions.append({
#                     "date": str(last_date + pd.Timedelta(days=i+1)),
#                     "predicted_close": round(pred_price, 2)
#                 })
                
#                 # Update sequence for next prediction
#                 next_features = current_sequence[-1].copy()
#                 next_features[0] = pred_scaled
#                 if len(feature_columns) > 1:
#                     next_features[4] = (next_features[4] * 4 + pred_scaled) / 5  # Update ma_5
#                     # Approximate open, high, low based on close
#                     next_features[1] = pred_scaled  # open
#                     next_features[2] = pred_scaled * 1.005  # high (approximation)
#                     next_features[3] = pred_scaled * 0.995  # low (approximation)
                
#                 current_sequence = np.roll(current_sequence, -1, axis=0)
#                 current_sequence[-1] = next_features
        
#         logger.info(f"Generated {days} predictions for XAU/USD.")
#         return predictions
        
#     except FileNotFoundError as e:
#         logger.error(f"Model files not found: {e}")
#         logger.info("Please train the model first using train_nn()")
#         return []
#     except Exception as e:
#         logger.error(f"Error in prediction: {e}")
#         import traceback
#         traceback.print_exc()
#         return []

# # --- Get Model Info (Unchanged) ---
# def get_model_info():
#     """Get information about the saved model"""
#     try:
#         if not os.path.exists(MODEL_PATH):
#             return "No model found. Train a model first."
        
#         checkpoint = torch.load(MODEL_PATH, map_location='cpu')
#         return {
#             "model_config": checkpoint.get('model_config', {}),
#             "feature_columns": checkpoint.get('feature_columns', []),
#             "sequence_length": checkpoint.get('sequence_length', 'Unknown'),
#             "pytorch_version": checkpoint.get('pytorch_version', 'Unknown'),
#             "file_size_mb": round(os.path.getsize(MODEL_PATH) / (1024*1024), 2)
#         }
#     except Exception as e:
#         return f"Error reading model info: {e}"