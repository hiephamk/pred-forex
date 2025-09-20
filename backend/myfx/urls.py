from django.urls import path
from .views import train_nn_view, forecast_nn

urlpatterns = [
    path("forex/train_nn/", train_nn_view),
    path("forex/forecast/", forecast_nn)
]