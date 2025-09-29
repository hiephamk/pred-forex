import torch
import pandas as pd
from .training_model import (ForexNN, train_nn, load_data)
from .models import PredictedForexData, ForexData


MODEL_PATH = "xauusd_nn.pt"
def predict_next_hours(hours=1):  # Changed default from 5 to 10
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
            
            PredictedForexData.objects.filter(date=iso_utc).delete()
             # Save prediction to PredictedForexData
            PredictedForexData.objects.create(
                date=iso_utc,
                predicted_close=round(pred_price.item(), 2)
            )
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