from django.urls import path
# from .views import train_nn_view, forecast_nn
from . import views

# urlpatterns = [
#     path("forex/train_nn/", train_nn_view),
#     path("forex/forecast/", forecast_nn)
# ]

urlpatterns = [
    path('forex/train/', views.train_nn_view, name='train_nn'),
    path('forex/forecast/', views.forecast_nn, name='forecast_nn'),
    path('forex/data-status/', views.data_status, name='data_status'),
    path('forex/recent-data/', views.recent_data, name='recent_data'),
    path('forex/fetch-data/', views.fetch_latest_data, name='fetch_latest_data'),
    path('forex/real-data/latest-data/', views.fetch_latest_data, name='latest_real_data'),
    path('forex/predictions/history/', views.PredictedForexDataView.as_view(), name='predictions_history'),
    path('forex/real_data/', views.ForexDataView.as_view(), name='forex_real_data'),
]