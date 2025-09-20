from django.shortcuts import render

# Create your views here.
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import ForexData
from .training_model import train_nn, predict_next

@api_view(["POST"])
def train_nn_view(request):
    model = train_nn()
    return Response({"status": "trained" if model else "no data"})
    
@api_view(["GET"])
def forecast_nn(request):
    days = int(request.GET.get("days", 5))
    preds = predict_next(days=days)
    return Response(preds)