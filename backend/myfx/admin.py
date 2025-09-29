from django.contrib import admin
from .models import ForexData, PredictedForexData
# Register your models here.
admin.site.register(ForexData)
admin.site.register(PredictedForexData)