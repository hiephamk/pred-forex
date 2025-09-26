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
]