
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import ForexData
from .training_model import (
    train_nn, 
    predict_next_hours, 
    get_hourly_predictions_for_today,
    load_data
)
from django.http import JsonResponse
from django.utils import timezone
import json

@api_view(["POST"])
def train_nn_view(request):
    """Train the neural network model"""
    try:
        # Get epochs from request data, default to 500
        epochs = 500
        if request.data:
            epochs = request.data.get('epochs', 500)
        
        print(f"Starting training with {epochs} epochs...")
        model = train_nn(epochs=epochs)
        
        if model:
            return Response({
                "status": "trained",
                "message": f"Model trained successfully with {epochs} epochs",
                "success": True
            }, status=status.HTTP_200_OK)
        else:
            return Response({
                "status": "no data",
                "message": "Failed to train model - insufficient data",
                "success": False
            }, status=status.HTTP_400_BAD_REQUEST)
            
    except Exception as e:
        return Response({
            "status": "error",
            "message": f"Training failed: {str(e)}",
            "success": False
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(["GET"])
def forecast_nn(request):
    """Get price predictions"""
    try:
        hours = int(request.GET.get('hours', 24))  # Default to 24 hours
        prediction_type = request.GET.get('type', 'next')  # 'next' or 'today'
        
        print(f"Generating {prediction_type} predictions for {hours} hours...")
        
        if prediction_type == 'today':
            predictions = get_hourly_predictions_for_today()
        else:
            predictions = predict_next_hours(hours=hours)
        
        if not predictions:
            return JsonResponse({
                "success": False,
                "message": "No predictions available. Model may need training or data may be insufficient.",
                "predictions": []
            }, status=400)
        
        return JsonResponse({
            "success": True,
            "predictions": predictions,
            "count": len(predictions),
            "type": prediction_type,
            "generated_at": timezone.now().isoformat()
        })
        
    except ValueError as e:
        return JsonResponse({
            "success": False,
            "error": f"Invalid parameter: {str(e)}",
            "predictions": []
        }, status=400)
        
    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": f"Prediction failed: {str(e)}",
            "predictions": []
        }, status=500)

@api_view(["GET"])
def data_status(request):
    """Get information about available data"""
    try:
        # Get data statistics
        total_records = ForexData.objects.count()
        latest_data = ForexData.objects.order_by('-date').first()
        oldest_data = ForexData.objects.order_by('date').first()
        
        # Calculate data coverage
        if latest_data and oldest_data:
            data_span = latest_data.date - oldest_data.date
            hours_covered = data_span.total_seconds() / 3600
        else:
            hours_covered = 0
        
        return Response({
            "success": True,
            "total_records": total_records,
            "latest_date": latest_data.date.isoformat() if latest_data else None,
            "oldest_date": oldest_data.date.isoformat() if oldest_data else None,
            "hours_covered": round(hours_covered, 1),
            "latest_price": float(latest_data.close) if latest_data else None,
            "current_time": timezone.now().isoformat()
        })
        
    except Exception as e:
        return Response({
            "success": False,
            "error": f"Failed to get data status: {str(e)}"
        }, status=500)

@api_view(["GET"])
def recent_data(request):
    """Get recent historical data"""
    try:
        hours = int(request.GET.get('hours', 24))  # Default to last 24 hours
        
        # Get recent data
        recent_data = ForexData.objects.order_by('-date')[:hours]
        
        data_list = []
        for item in recent_data:
            data_list.append({
                'datetime': item.date.isoformat(),
                'open': float(item.open),
                'high': float(item.high),
                'low': float(item.low),
                'close': float(item.close),
            })
        
        # Reverse to get chronological order
        data_list.reverse()
        
        return JsonResponse({
            "success": True,
            "data": data_list,
            "count": len(data_list),
            "hours_requested": hours
        })
        
    except ValueError:
        return JsonResponse({
            "success": False,
            "error": "Invalid hours parameter"
        }, status=400)
        
    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": f"Failed to get recent data: {str(e)}"
        }, status=500)

@api_view(["POST"])
def fetch_latest_data(request):
    """Manually trigger fetching of latest data"""
    try:
        from .utils import fetch_xauusd
        
        # Fetch latest data
        fetch_xauusd(latest_only=True)
        
        # Get updated count
        total_records = ForexData.objects.count()
        latest_data = ForexData.objects.order_by('-date').first()
        
        return Response({
            "success": True,
            "message": "Latest data fetched successfully",
            "total_records": total_records,
            "latest_date": latest_data.date.isoformat() if latest_data else None,
            "latest_price": float(latest_data.close) if latest_data else None
        })
        
    except Exception as e:
        return Response({
            "success": False,
            "error": f"Failed to fetch data: {str(e)}"
        }, status=500)