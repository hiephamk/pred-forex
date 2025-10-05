"""
Prediction functions for XAU/USD forex forecasting
Separate from training_model.py for better organization
"""
import torch
import pandas as pd
import numpy as np
from .models import ForexData, PredictedForexData
from .training_model import (
    ForexLSTM,
    create_advanced_features,
    load_scalers,
    train_lstm,
    MODEL_PATH
)
from datetime import datetime as dt


def predict_next_hours(hours=24, sequence_length=60):
    """
    Predict prices for the next N hours starting from the next full hour in UTC.
    Returns a list of dicts with ISO 8601 datetimes (UTC, 'Z') and predicted_close.
    
    Args:
        hours: Number of hours to predict (default: 24)
        sequence_length: Sequence length used in model (default: 60)
    
    Returns:
        List of predictions: [{"datetime": "2025-10-05T12:00:00Z", "predicted_close": 2650.50}, ...]
    """
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
        input_features = checkpoint.get('input_features', 33)
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
    if len(X_scaled) < sequence_length:
        print(f"Insufficient data for prediction. Need at least {sequence_length} records, have {len(X_scaled)}")
        return []
    
    last_sequence = X_scaled[-sequence_length:]
    
    # Use UTC for predictions
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
            
            # Format datetime in ISO 8601 UTC (Z)
            iso_utc = pred_time_utc.isoformat().replace("+00:00", "Z")
            
            # Save to database (remove duplicates)
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
            # In production, you'd want to recalculate all features properly
            last_sequence = np.roll(last_sequence, -1, axis=0)
    
    return predictions


def get_hourly_predictions_for_today():
    """
    Get predictions for the remaining hours today in UTC.
    Returns predictions from current hour to end of day (23:00 UTC).
    
    Returns:
        List of predictions for remaining hours today
    """
    current_time_utc = pd.Timestamp.now(tz="UTC").replace(minute=0, second=0, microsecond=0)
    
    # Calculate hours remaining in the day (until 23:00 UTC)
    end_of_day = current_time_utc.replace(hour=23, minute=0, second=0, microsecond=0)
    
    if current_time_utc >= end_of_day:
        # If it's already past 23:00, predict for next day
        hours = 24
    else:
        # Calculate remaining hours including the next full hour
        hours = (end_of_day - current_time_utc).seconds // 3600 + 1
    
    print(f"Generating predictions for {hours} remaining hours today")
    return predict_next_hours(hours=hours)


def get_prediction_stats():
    """
    Get statistics about current predictions.
    Useful for monitoring model performance.
    
    Returns:
        Dict with prediction statistics
    """
    predictions = PredictedForexData.objects.order_by('date')
    
    if not predictions.exists():
        return {
            "total_predictions": 0,
            "message": "No predictions available"
        }
    
    prediction_values = [float(p.predicted_close) for p in predictions]
    
    return {
        "total_predictions": len(prediction_values),
        "earliest_prediction": predictions.first().date.isoformat(),
        "latest_prediction": predictions.last().date.isoformat(),
        "min_predicted_price": min(prediction_values),
        "max_predicted_price": max(prediction_values),
        "avg_predicted_price": sum(prediction_values) / len(prediction_values),
        "prediction_range": max(prediction_values) - min(prediction_values)
    }


def compare_predictions_with_actual(hours_back=24):
    """
    Compare predictions with actual prices for model evaluation.
    
    Args:
        hours_back: Number of hours to look back for comparison
    
    Returns:
        Dict with comparison metrics
    """
    cutoff_time = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours_back)
    
    # Get predictions that should have actual data now
    old_predictions = PredictedForexData.objects.filter(
        date__gte=cutoff_time,
        date__lte=pd.Timestamp.now(tz="UTC")
    ).order_by('date')
    
    comparisons = []
    errors = []
    
    for pred in old_predictions:
        # Find matching actual data
        actual_data = ForexData.objects.filter(date=pred.date).first()
        
        if actual_data:
            predicted = float(pred.predicted_close)
            actual = float(actual_data.close)
            error = abs(predicted - actual)
            percent_error = (error / actual) * 100
            
            comparisons.append({
                "datetime": pred.date.isoformat(),
                "predicted": predicted,
                "actual": actual,
                "error": round(error, 2),
                "percent_error": round(percent_error, 2)
            })
            errors.append(error)
    
    if not errors:
        return {
            "message": "No matching predictions and actual data found for comparison",
            "comparisons": []
        }
    
    # Calculate metrics
    mae = sum(errors) / len(errors)  # Mean Absolute Error
    rmse = (sum([e**2 for e in errors]) / len(errors)) ** 0.5  # Root Mean Square Error
    
    return {
        "total_comparisons": len(comparisons),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "average_percent_error": round(sum([c['percent_error'] for c in comparisons]) / len(comparisons), 2),
        "comparisons": comparisons
    }