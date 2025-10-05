import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from .models import ForexData, PredictedForexData
from .utils import fetch_xauusd
import pytz
from datetime import datetime as dt
import json

MODEL_PATH = "xauusd_lstm.pt"
SCALER_PATH = "xauusd_scaler.json"

class MinMaxScaler:
    """Simple MinMaxScaler implementation without sklearn"""
    def __init__(self):
        self.min_vals = None
        self.max_vals = None
    
    def fit(self, data):
        self.min_vals = np.min(data, axis=0)
        self.max_vals = np.max(data, axis=0)
        return self
    
    def transform(self, data):
        if self.min_vals is None or self.max_vals is None:
            raise ValueError("Scaler must be fitted before transform")
        
        # Avoid division by zero
        range_vals = self.max_vals - self.min_vals
        range_vals[range_vals == 0] = 1
        
        return (data - self.min_vals) / range_vals
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data):
        if self.min_vals is None or self.max_vals is None:
            raise ValueError("Scaler must be fitted before inverse_transform")
        
        range_vals = self.max_vals - self.min_vals
        range_vals[range_vals == 0] = 1
        
        return data * range_vals + self.min_vals
    
    def to_dict(self):
        return {
            'min_vals': self.min_vals.tolist() if self.min_vals is not None else None,
            'max_vals': self.max_vals.tolist() if self.max_vals is not None else None
        }
    
    @classmethod
    def from_dict(cls, data):
        scaler = cls()
        if data['min_vals'] is not None:
            scaler.min_vals = np.array(data['min_vals'])
        if data['max_vals'] is not None:
            scaler.max_vals = np.array(data['max_vals'])
        return scaler


class ForexLSTM(nn.Module):
    """LSTM model for forex time series prediction"""
    def __init__(self, input_features=10, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # x shape: (batch, sequence, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Final prediction
        output = self.fc(context)
        return output


def create_advanced_features(df):
    """Create comprehensive technical indicators for forex"""
    df = df.copy()
    
    # Ensure datetime is in UTC
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('UTC')
    
    # Time-based features
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    
    # Price features
    df['hl_range'] = df['high'] - df['low']
    df['oc_range'] = df['close'] - df['open']
    
    # Moving averages
    df['sma_12'] = df['close'].rolling(window=12, min_periods=1).mean()
    df['sma_24'] = df['close'].rolling(window=24, min_periods=1).mean()
    df['sma_48'] = df['close'].rolling(window=48, min_periods=1).mean()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_24'] = df['close'].ewm(span=24, adjust=False).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_24']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20, min_periods=1).mean()
    bb_std = df['close'].rolling(window=20, min_periods=1).std()
    df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
    df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    
    # Volatility
    df['volatility'] = df['close'].rolling(window=24, min_periods=1).std()
    df['price_change'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / (df['close'].shift(1) + 1e-10))
    
    # Momentum
    df['momentum_12'] = df['close'] - df['close'].shift(12)
    df['momentum_24'] = df['close'] - df['close'].shift(24)
    df['rate_of_change'] = ((df['close'] - df['close'].shift(12)) / 
                             (df['close'].shift(12) + 1e-10)) * 100
    
    # Lag features
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
    
    # Fill NaN values
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    return df


def create_sequences(X, y, sequence_length=60):
    """Create sequences for LSTM training"""
    X_seq, y_seq = [], []
    
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    
    return np.array(X_seq), np.array(y_seq)


def save_scalers(scaler_X, scaler_y):
    """Save scalers to JSON file"""
    data = {
        'X': scaler_X.to_dict(),
        'y': scaler_y.to_dict()
    }
    with open(SCALER_PATH, 'w') as f:
        json.dump(data, f)


def load_scalers():
    """Load scalers from JSON file"""
    try:
        with open(SCALER_PATH, 'r') as f:
            data = json.load(f)
        return MinMaxScaler.from_dict(data['X']), MinMaxScaler.from_dict(data['y'])
    except FileNotFoundError:
        return None, None


def load_data(fetch_if_empty=True, fetch_latest=False, sequence_length=60):
    """Load and prepare data for LSTM training"""
    if not ForexData.objects.exists() and fetch_if_empty:
        print("No data found, fetching initial data...")
        fetch_xauusd(latest_only=False)
    
    if fetch_latest:
        print("Fetching latest data...")
        fetch_xauusd(latest_only=True)
    
    data = ForexData.objects.order_by('date').values('date', 'open', 'high', 'low', 'close')
    if not data:
        print("No data available after fetching")
        return None, None, None, None, None
    
    df = pd.DataFrame(data)
    df = create_advanced_features(df)
    
    # Select comprehensive features
    feature_cols = [
        'hour', 'day_of_week', 'day_of_month', 'month',
        'open', 'high', 'low', 'hl_range', 'oc_range',
        'sma_12', 'sma_24', 'sma_48', 'ema_12', 'ema_24',
        'macd', 'macd_signal', 'macd_diff',
        'rsi', 'bb_width', 'bb_position',
        'volatility', 'price_change', 'log_return',
        'momentum_12', 'momentum_24', 'rate_of_change',
        'close_lag_1', 'close_lag_2', 'close_lag_3', 
        'close_lag_6', 'close_lag_12', 'close_lag_24'
    ]
    
    X = df[feature_cols].values
    y = df['close'].values.reshape(-1, 1)
    
    # Normalize features using custom MinMaxScaler
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Save scalers
    save_scalers(scaler_X, scaler_y)
    
    # Create sequences
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, sequence_length)
    
    # Convert to tensors
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32)
    
    latest_time_utc = df['date'].iloc[-1]
    
    print(f"Data loaded: {len(df)} records")
    print(f"Sequences created: {len(X_seq)}")
    print(f"Latest data point (UTC): {latest_time_utc}")
    print(f"Price range - min: {y.min():.2f}, max: {y.max():.2f}")
    
    return X_tensor, y_tensor, df, scaler_X, scaler_y


# Alias for backward compatibility
def train_nn(epochs=200, sequence_length=60, batch_size=32):
    """Train neural network model (LSTM) - backward compatible name"""
    return train_lstm(epochs, sequence_length, batch_size)


def train_lstm(epochs=200, sequence_length=60, batch_size=32):
    """Train LSTM model with mini-batch training"""
    X, y, df, scaler_X, scaler_y = load_data(
        fetch_if_empty=True, 
        fetch_latest=True, 
        sequence_length=sequence_length
    )
    
    if X is None or y is None:
        print("Failed to load data for training")
        return None
    
    # Split into train and validation (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    model = ForexLSTM(input_features=X.shape[2], hidden_size=128, num_layers=2, dropout=0.3)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )
    
    print("Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]
            
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val).item()
        
        train_loss /= (len(X_train) // batch_size + 1)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'input_features': X.shape[2]
            }, MODEL_PATH)
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Best Val: {best_val_loss:.6f}")
        
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    return model


def predict_next_hours(hours=24, sequence_length=60):
    """Predict prices for the next N hours using LSTM"""
    # Load data
    data = ForexData.objects.order_by('date').values('date', 'open', 'high', 'low', 'close')
    if not data:
        print("No data available for prediction")
        return []
    
    df = pd.DataFrame(data)
    df = create_advanced_features(df)
    
    # Load scalers
    scaler_X, scaler_y = load_scalers()
    if scaler_X is None or scaler_y is None:
        print("Scalers not found. Please train the model first.")
        return []
    
    # Load model
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        input_features = checkpoint.get('input_features', 33)  # Default to 33 features
        model = ForexLSTM(input_features=input_features, hidden_size=128, num_layers=2, dropout=0.3)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully (trained epoch: {checkpoint['epoch']})")
    except FileNotFoundError:
        print("Model not found. Training a new model...")
        model = train_lstm(sequence_length=sequence_length)
        if model is None:
            return []
    
    model.eval()
    
    # Prepare features
    feature_cols = [
        'hour', 'day_of_week', 'day_of_month', 'month',
        'open', 'high', 'low', 'hl_range', 'oc_range',
        'sma_12', 'sma_24', 'sma_48', 'ema_12', 'ema_24',
        'macd', 'macd_signal', 'macd_diff',
        'rsi', 'bb_width', 'bb_position',
        'volatility', 'price_change', 'log_return',
        'momentum_12', 'momentum_24', 'rate_of_change',
        'close_lag_1', 'close_lag_2', 'close_lag_3', 
        'close_lag_6', 'close_lag_12', 'close_lag_24'
    ]
    
    X = df[feature_cols].values
    X_scaled = scaler_X.transform(X)
    
    # Get last sequence
    last_sequence = X_scaled[-sequence_length:]
    
    current_time_utc = pd.Timestamp.now(tz="UTC").replace(minute=0, second=0, microsecond=0)
    start_time_utc = current_time_utc + pd.Timedelta(hours=1)
    
    predictions = []
    
    with torch.no_grad():
        for i in range(hours):
            pred_time_utc = start_time_utc + pd.Timedelta(hours=i)
            
            # Prepare input sequence
            input_seq = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0)
            
            # Predict
            pred_scaled = model(input_seq)
            pred_price = scaler_y.inverse_transform(pred_scaled.numpy())[0][0]
            
            # Format datetime
            iso_utc = pred_time_utc.isoformat().replace("+00:00", "Z")
            
            # Save to database
            PredictedForexData.objects.filter(date=iso_utc).delete()
            PredictedForexData.objects.create(
                date=iso_utc,
                predicted_close=round(pred_price, 2)
            )
            
            predictions.append({
                "datetime": iso_utc,
                "predicted_close": round(pred_price, 2)
            })
            
            # Update sequence for next prediction (simplified)
            last_sequence = np.roll(last_sequence, -1, axis=0)
    
    return predictions