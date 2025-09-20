
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

### version 2
"""
version-2
"""
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from .models import ForexData
# from .utils import fetch_xauusd

# MODEL_PATH = "xauusd_nn.pt"

# class ForexNN(nn.Module):
#     def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
#         super(ForexNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
#         self.fc = nn.Linear(hidden_size, output_size)
#         self.relu = nn.ReLU()
    
#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[:, -1, :])  # Take the last time step's output
#         return out

# def load_data(fetch_if_empty=True, window_size=30, normalize=True):
#     """
#     Load XAU/USD data from DB and prepare sequences for LSTM.
#     - fetch_if_empty: automatically fetch data if table is empty
#     - window_size: number of past days to use for prediction
#     - normalize: scale close prices to 0-1 for training
#     """
#     if ForexData.objects.count() == 0 and fetch_if_empty:
#         fetch_xauusd()

#     qs = ForexData.objects.all().order_by("date")
#     df = pd.DataFrame(list(qs.values("date", "close")))
#     if df.empty:
#         return None, None, None, None, None

#     df["date"] = pd.to_datetime(df["date"])
    
#     # Normalize close prices
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(df["close"].values.reshape(-1, 1)) if normalize else df["close"].values.reshape(-1, 1)

#     # Create sequences
#     X, y = [], []
#     for i in range(len(scaled_data) - window_size):
#         X.append(scaled_data[i:i + window_size])
#         y.append(scaled_data[i + window_size])

#     X = np.array(X)
#     y = np.array(y)

#     # Convert to tensors
#     X = torch.tensor(X, dtype=torch.float32)
#     y = torch.tensor(y, dtype=torch.float32)

#     return X, y, df, scaler if normalize else None, window_size

# def train_nn(epochs=200, lr=0.001, train_split=0.8, patience=20):
#     X, y, df, scaler, window_size = load_data()
#     if X is None:
#         print("No data to train")
#         return None

#     # Train-test split (time-series aware)
#     train_size = int(len(X) * train_split)
#     X_train, X_val = X[:train_size], X[train_size:]
#     y_train, y_val = y[:train_size], y[train_size:]

#     model = ForexNN(input_size=1, hidden_size=64, num_layers=2)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # L2 regularization
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

#     best_val_loss = float('inf')
#     early_stop_counter = 0

#     for epoch in range(epochs):
#         model.train()
#         optimizer.zero_grad()
#         outputs = model(X_train)
#         loss = criterion(outputs, y_train)
#         loss.backward()
#         optimizer.step()

#         # Validation
#         model.eval()
#         with torch.no_grad():
#             val_outputs = model(X_val)
#             val_loss = criterion(val_outputs, y_val)

#         scheduler.step(val_loss)

#         if epoch % 10 == 0:
#             print(f"Epoch {epoch}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

#         # Early stopping
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             early_stop_counter = 0
#             torch.save(model.state_dict(), MODEL_PATH)
#         else:
#             early_stop_counter += 1
#             if early_stop_counter >= patience:
#                 print(f"Early stopping at epoch {epoch}")
#                 break

#     print(f"Neuro Network saved to {MODEL_PATH}")
#     return model

# def predict_next(days=5, normalize=True):
#     X, y, df, scaler, window_size = load_data(fetch_if_empty=True, normalize=normalize)
#     if X is None:
#         return []

#     model = ForexNN(input_size=1, hidden_size=64, num_layers=2)
#     model.load_state_dict(torch.load(MODEL_PATH))
#     model.eval()

#     last_date = df["date"].iloc[-1]
#     predictions = []

#     # Start with the last window of actual data
#     current_window = df["close"].values[-window_size:].reshape(-1, 1)
#     if normalize and scaler is not None:
#         current_window = scaler.transform(current_window)
#     current_window = current_window.reshape(1, window_size, 1)
#     current_window = torch.tensor(current_window, dtype=torch.float32)

#     for i in range(days):
#         with torch.no_grad():
#             pred = model(current_window)
#             pred_denorm = scaler.inverse_transform(pred.numpy())[0][0] if normalize and scaler is not None else pred.numpy()[0][0]
#             predictions.append({
#                 "date": str(last_date + pd.Timedelta(days=i + 1)),
#                 "predicted_close": round(pred_denorm, 2)
#             })

#         # Update window with the new prediction (autoregressive)
#         new_pred = pred.numpy().reshape(1, 1, 1)
#         current_window = torch.cat((current_window[:, 1:, :], torch.tensor(new_pred)), dim=1)

#     return predictions

"""
version-3
"""

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# import numpy as np
# import os
# import pickle
# from django.conf import settings
# from .models import ForexData
# from .utils import fetch_xauusd
# import warnings
# warnings.filterwarnings('ignore')

# # Use Django's media root for model storage
# MODEL_DIR = os.path.join(settings.MEDIA_ROOT, 'forex_models')
# os.makedirs(MODEL_DIR, exist_ok=True)
# MODEL_PATH = os.path.join(MODEL_DIR, "xauusd_nn.pt")
# SCALER_PATH = os.path.join(MODEL_DIR, "xauusd_scaler.pkl")

# class ForexNN(nn.Module):
#     def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.2):
#         super(ForexNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         # LSTM layers for sequence modeling
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
#                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
#         # Fully connected layers with dropout
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
#         # LSTM forward pass
#         lstm_out, _ = self.lstm(x)
#         # Use the last output from LSTM
#         last_output = lstm_out[:, -1, :]
#         # Pass through fully connected layers
#         output = self.fc_layers(last_output)
#         return output

# def create_technical_indicators(df):
#     """Create technical indicators as features using Django QuerySet data"""
#     df = df.copy()
    
#     # Ensure we have required columns
#     required_cols = ['close', 'open', 'high', 'low', 'volume']
#     missing_cols = [col for col in required_cols if col not in df.columns]
    
#     # If missing OHLV data, create basic approximations
#     if missing_cols:
#         if 'open' not in df.columns:
#             df['open'] = df['close'].shift(1).fillna(df['close'])
#         if 'high' not in df.columns:
#             df['high'] = df['close'] * 1.001  # Small approximation
#         if 'low' not in df.columns:
#             df['low'] = df['close'] * 0.999
#         if 'volume' not in df.columns:
#             df['volume'] = 1000  # Default volume
    
#     # Moving averages
#     df['ma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
#     df['ma_10'] = df['close'].rolling(window=10, min_periods=1).mean()
#     df['ma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
    
#     # RSI (simplified version)
#     def calculate_rsi(prices, period=14):
#         delta = prices.diff()
#         gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
#         loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
#         rs = gain / (loss + 1e-8)  # Add small epsilon to avoid division by zero
#         return 100 - (100 / (1 + rs))
    
#     df['rsi'] = calculate_rsi(df['close'])
    
#     # Price momentum
#     df['momentum'] = df['close'].pct_change(periods=5).fillna(0)
    
#     # Bollinger Bands position (simplified)
#     bb_period = 20
#     df['bb_middle'] = df['close'].rolling(window=bb_period, min_periods=1).mean()
#     bb_std = df['close'].rolling(window=bb_period, min_periods=1).std()
#     df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
#     df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
#     # Avoid division by zero in bb_position
#     bb_range = df['bb_upper'] - df['bb_lower']
#     df['bb_position'] = np.where(bb_range > 0, 
#                                 (df['close'] - df['bb_lower']) / bb_range, 
#                                 0.5)  # Default to middle position
    
#     # Fill any remaining NaN values
#     df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
#     return df

# class SimpleScaler:
#     """Simple MinMax scaler to avoid sklearn dependency"""
#     def __init__(self):
#         self.data_min_ = None
#         self.data_max_ = None
#         self.data_range_ = None
        
#     def fit(self, X):
#         self.data_min_ = np.min(X, axis=0)
#         self.data_max_ = np.max(X, axis=0)
#         self.data_range_ = self.data_max_ - self.data_min_
#         # Avoid division by zero
#         self.data_range_[self.data_range_ == 0] = 1.0
#         return self
        
#     def transform(self, X):
#         return (X - self.data_min_) / self.data_range_
        
#     def fit_transform(self, X):
#         return self.fit(X).transform(X)
        
#     def inverse_transform(self, X):
#         return X * self.data_range_ + self.data_min_

# def create_sequences(data, sequence_length=15):
#     """Create sequences for LSTM input"""
#     X, y = [], []
#     for i in range(sequence_length, len(data)):
#         X.append(data[i-sequence_length:i])
#         y.append(data[i, 0])  # Close price is first column
#     return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# def load_data(sequence_length=15, test_split=0.2):
#     """Load and prepare data from Django ForexData model"""
    
#     # Fetch data if needed
#     if ForexData.objects.count() == 0:
#         try:
#             fetch_xauusd()
#         except Exception as e:
#             print(f"Could not fetch data: {e}")
#             return None, None, None, None, None, None
    
#     # Get data from Django model
#     qs = ForexData.objects.all().order_by("date")
    
#     # Handle different possible field names in your ForexData model
#     field_names = [f.name for f in ForexData._meta.get_fields()]
    
#     # Build values list based on available fields
#     values_list = ["date", "close"]
#     optional_fields = ["open", "high", "low", "volume"]
#     for field in optional_fields:
#         if field in field_names:
#             values_list.append(field)
    
#     df = pd.DataFrame(list(qs.values(*values_list)))
    
#     if df.empty or len(df) < sequence_length + 10:
#         print(f"Not enough data for training. Need at least {sequence_length + 10} records, got {len(df)}")
#         return None, None, None, None, None, None
        
#     df["date"] = pd.to_datetime(df["date"])
#     df = df.sort_values("date").reset_index(drop=True)
    
#     # Create technical indicators
#     df = create_technical_indicators(df)
    
#     # Select features for model (ensure they exist)
#     available_features = ['close', 'ma_5', 'ma_10', 'rsi', 'bb_position']
#     feature_columns = [col for col in available_features if col in df.columns]
    
#     if len(feature_columns) < 2:
#         print("Not enough features available")
#         return None, None, None, None, None, None
    
#     print(f"Using features: {feature_columns}")
    
#     # Prepare feature matrix
#     features = df[feature_columns].values.astype(np.float32)
    
#     # Scale features
#     scaler = SimpleScaler()
#     features_scaled = scaler.fit_transform(features)
    
#     # Create sequences
#     X, y = create_sequences(features_scaled, sequence_length)
    
#     if len(X) == 0:
#         print("No sequences could be created")
#         return None, None, None, None, None, None
    
#     # Split into train/test
#     split_idx = int(len(X) * (1 - test_split))
#     X_train, X_test = X[:split_idx], X[split_idx:]
#     y_train, y_test = y[:split_idx], y[split_idx:]
    
#     # Convert to tensors
#     X_train = torch.FloatTensor(X_train)
#     X_test = torch.FloatTensor(X_test)
#     y_train = torch.FloatTensor(y_train).unsqueeze(1)
#     y_test = torch.FloatTensor(y_test).unsqueeze(1)
    
#     return X_train, X_test, y_train, y_test, scaler, feature_columns

# def train_nn(epochs=150, lr=0.001, sequence_length=15):
#     """Train the neural network with Django integration"""
    
#     print("Loading and preparing data...")
#     data = load_data(sequence_length=sequence_length)
    
#     if data[0] is None:
#         print("No data available for training")
#         return None
        
#     X_train, X_test, y_train, y_test, scaler, feature_columns = data
    
#     print(f"Training data shape: {X_train.shape}")
#     print(f"Test data shape: {X_test.shape}")
#     print(f"Features: {feature_columns}")
    
#     # Initialize model
#     model = ForexNN(input_size=len(feature_columns))
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)
    
#     # Training variables
#     best_val_loss = float('inf')
#     patience_counter = 0
#     patience = 30
    
#     for epoch in range(epochs):
#         # Training
#         model.train()
#         optimizer.zero_grad()
        
#         train_outputs = model(X_train)
#         train_loss = criterion(train_outputs, y_train)
        
#         train_loss.backward()
#         # Gradient clipping
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
        
#         # Validation
#         model.eval()
#         with torch.no_grad():
#             val_outputs = model(X_test)
#             val_loss = criterion(val_outputs, y_test)
            
#         # Learning rate scheduling
#         scheduler.step(val_loss)
        
#         # Early stopping
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             patience_counter = 0
#             # Save best model with metadata
#             torch.save({
#                 'model_state_dict': model.state_dict(),
#                 'feature_columns': feature_columns,
#                 'sequence_length': sequence_length,
#                 'input_size': len(feature_columns)
#             }, MODEL_PATH)
#             # Save scaler separately
#             with open(SCALER_PATH, 'wb') as f:
#                 pickle.dump(scaler, f)
#         else:
#             patience_counter += 1
            
#         if epoch % 20 == 0:
#             print(f"Epoch {epoch}/{epochs}")
#             print(f"Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
#             print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")
#             print("-" * 50)
            
#         if patience_counter >= patience:
#             print(f"Early stopping at epoch {epoch}")
#             break
    
#     # Calculate final metrics
#     model.eval()
#     with torch.no_grad():
#         train_pred = model(X_train)
#         test_pred = model(X_test)
        
#         train_mse = criterion(train_pred, y_train).item()
#         test_mse = criterion(test_pred, y_test).item()
        
#         print(f"\nFinal Results:")
#         print(f"Train MSE: {train_mse:.6f}")
#         print(f"Test MSE: {test_mse:.6f}")
#         print(f"Neural Network saved to {MODEL_PATH}")
    
#     return model

# def predict_next(days=5, sequence_length=15):
#     """Make predictions using the trained model"""
    
#     try:
#         # Load model
#         if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
#             print("Model or scaler not found. Please train the model first.")
#             return []
            
#         checkpoint = torch.load(MODEL_PATH, map_location='cpu')
#         feature_columns = checkpoint['feature_columns']
#         saved_sequence_length = checkpoint.get('sequence_length', sequence_length)
#         input_size = checkpoint['input_size']
        
#         # Load scaler
#         with open(SCALER_PATH, 'rb') as f:
#             scaler = pickle.load(f)
        
#         # Initialize and load model
#         model = ForexNN(input_size=input_size)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         model.eval()
        
#         # Get latest data from Django model
#         qs = ForexData.objects.all().order_by("date")
        
#         # Get field names and build values list
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
        
#         df["date"] = pd.to_datetime(df["date"])
#         df = df.sort_values("date").reset_index(drop=True)
        
#         # Create technical indicators
#         df = create_technical_indicators(df)
        
#         # Get last sequence
#         features = df[feature_columns].values.astype(np.float32)
#         features_scaled = scaler.transform(features)
#         last_sequence = features_scaled[-saved_sequence_length:]
        
#         predictions = []
#         current_sequence = last_sequence.copy()
#         last_date = df["date"].iloc[-1]
        
#         with torch.no_grad():
#             for i in range(days):
#                 # Prepare input tensor
#                 input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0)
                
#                 # Make prediction (normalized)
#                 pred_scaled = model(input_tensor).item()
                
#                 # Denormalize prediction
#                 # Create dummy array for inverse transform
#                 dummy_features = np.zeros((1, len(feature_columns)))
#                 dummy_features[0, 0] = pred_scaled
#                 pred_price = scaler.inverse_transform(dummy_features)[0, 0]
                
#                 # Add to predictions
#                 future_date = last_date + pd.Timedelta(days=i+1)
#                 predictions.append({
#                     "date": str(future_date.date()),
#                     "predicted_close": round(pred_price, 2)
#                 })
                
#                 # Update sequence for next prediction (simplified approach)
#                 next_features = current_sequence[-1].copy()
#                 next_features[0] = pred_scaled  # Update close price
                
#                 # Update moving averages approximately
#                 if len(feature_columns) > 1:
#                     next_features[1] = (next_features[1] * 4 + pred_scaled) / 5  # ma_5 approx
#                 if len(feature_columns) > 2:
#                     next_features[2] = (next_features[2] * 9 + pred_scaled) / 10  # ma_10 approx
                
#                 # Shift sequence and add new prediction
#                 current_sequence = np.roll(current_sequence, -1, axis=0)
#                 current_sequence[-1] = next_features
        
#         return predictions
        
#     except Exception as e:
#         print(f"Error in prediction: {e}")
#         import traceback
#         traceback.print_exc()
#         return []

# # Backwards compatibility functions
# def load_and_prepare_data(*args, **kwargs):
#     """Backwards compatibility wrapper"""
#     return load_data(*args, **kwargs)

# def train_enhanced_model(*args, **kwargs):
#     """Backwards compatibility wrapper"""
#     return train_nn(*args, **kwargs)

# def predict_next_enhanced(*args, **kwargs):
#     """Backwards compatibility wrapper"""
#     return predict_next(*args, **kwargs)
