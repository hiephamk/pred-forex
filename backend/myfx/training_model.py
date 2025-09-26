import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from .models import ForexData
from .utils import fetch_xauusd
import pytz
import numpy as np
from datetime import datetime as dt

MODEL_PATH = "xauusd_nn.pt"

class ForexNN(nn.Module):
    def __init__(self, input_features=5):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

def create_features(df):
    """Create features for the model"""
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
    
    # Technical indicators
    df['sma_24'] = df['close'].rolling(window=24, min_periods=1).mean()
    df['price_change'] = df['close'].pct_change().fillna(0)
    df['volatility'] = df['close'].rolling(window=24, min_periods=1).std().fillna(0)
    
    # Lag features
    df['close_lag_1'] = df['close'].shift(1).fillna(df['close'].iloc[0])
    df['close_lag_24'] = df['close'].shift(24).fillna(df['close'].iloc[0])
    
    # Normalize time features
    df['hour_norm'] = df['hour'] / 23.0
    df['day_of_week_norm'] = df['day_of_week'] / 6.0
    df['day_of_month_norm'] = (df['day_of_month'] - 1) / 30.0
    df['month_norm'] = (df['month'] - 1) / 11.0
    
    return df

def load_data(fetch_if_empty=True, fetch_latest=False, normalize=True):
    if not ForexData.objects.exists() and fetch_if_empty:
        print("No data found, fetching initial data...")
        fetch_xauusd(latest_only=False)
    
    if fetch_latest:
        print("Fetching latest data...")
        fetch_xauusd(latest_only=True)

    # Include all necessary fields from ForexData
    data = ForexData.objects.order_by('date').values('date', 'open', 'high', 'low', 'close')
    if not data:
        print("No data available after fetching")
        return None, None, None, None, None

    df = pd.DataFrame(data)
    df = create_features(df)
    
    # Select features for training (consistent with prediction)
    feature_cols = ['hour_norm', 'day_of_week_norm', 'close_lag_1', 'sma_24', 'volatility']
    
    # Handle potential NaN values
    df[feature_cols] = df[feature_cols].fillna(0)
    
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df['close'].values, dtype=torch.float32).unsqueeze(1)

    if normalize:
        y_min, y_max = y.min(), y.max()
        if y_max != y_min:
            y_norm = (y - y_min) / (y_max - y_min)
        else:
            y_norm = y
            print("Warning: y_max equals y_min, normalization skipped")
        print(f"Price range - min: {y_min.item():.2f}, max: {y_max.item():.2f}")
    else:
        y_norm = y
        y_min, y_max = None, None

    latest_time_utc = df['date'].iloc[-1]
    
    print(f"Data loaded: {len(df)} records")
    print(f"Latest data point (UTC): {latest_time_utc}")
    
    return X, y_norm, df, y_min, y_max

def train_nn(epochs=500):
    X, y, df, _, _ = load_data(fetch_if_empty=True, fetch_latest=True, normalize=True)
    if X is None or y is None:
        print("Failed to load data for training")
        return None

    model = ForexNN(input_features=X.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)

    print("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        pred = model(X)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), MODEL_PATH)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    print(f"Training completed. Best loss: {best_loss:.6f}")
    return model

def predict_next_hours(hours=10):  # Changed default from 5 to 10
    """
    Predict prices for the next N hours starting from the next full hour in UTC.
    Returns a list of dicts with ISO 8601 datetimes (UTC, 'Z') and predicted_close.
    """
    # Load data and model
    X, _, df, y_min, y_max = load_data(fetch_latest=True, normalize=True)
    if X is None or df is None:
        print("Failed to load data for prediction")
        return []

    model = ForexNN(input_features=X.shape[1])
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        print("Model loaded successfully")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Model loading failed: {e}, training a new model...")
        model = train_nn()
        if model is None:
            return []

    model.eval()

    # Use UTC for predictions
    current_time_utc = pd.Timestamp.now(tz="UTC").replace(minute=0, second=0, microsecond=0)
    start_time_utc = current_time_utc + pd.Timedelta(hours=1)

    # Use last known data point to initialize features
    last_row = df.iloc[-1].copy()

    predictions = []

    with torch.no_grad():
        for i in range(hours):
            pred_time_utc = start_time_utc + pd.Timedelta(hours=i)

            # Features for prediction (consistent with training)
            hour_norm = pred_time_utc.hour / 23.0
            day_of_week_norm = pred_time_utc.dayofweek / 6.0
            close_lag_1 = last_row['close']
            sma_24 = last_row['sma_24']
            volatility = last_row['volatility']

            features = torch.tensor([[hour_norm, day_of_week_norm, close_lag_1, sma_24, volatility]], 
                                  dtype=torch.float32)
            pred = model(features)

            # Denormalize prediction
            if y_min is not None and y_max is not None and y_max != y_min:
                pred_price = pred * (y_max - y_min) + y_min
            else:
                pred_price = pred

            # Format datetime in ISO 8601 UTC (Z)
            iso_utc = pred_time_utc.isoformat().replace("+00:00", "Z")

            predictions.append({
                "datetime": iso_utc,
                "predicted_close": round(pred_price.item(), 2)
            })

            # Update last_row for next iteration
            last_row['close'] = pred_price.item()
            # Update SMA using exponential moving average approximation
            last_row['sma_24'] = (last_row['close'] * 0.04167 + last_row['sma_24'] * 0.95833)
            # Update volatility (simplified)
            last_row['volatility'] = last_row['volatility'] * 0.95833  # Decay previous volatility

    return predictions

def get_hourly_predictions_for_today():
    """
    Get predictions for the remaining hours today in UTC.
    """
    current_time_utc = pd.Timestamp.now(tz="UTC").replace(minute=0, second=0, microsecond=0)
    # Fixed hours to reach 2025-09-27 03:00:00 UTC from 18:00 UTC
    hours = 10
    return predict_next_hours(hours=hours)