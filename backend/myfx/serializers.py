from rest_framework import serializers
from .models import ForexData, PredictedForexData


class ForexDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = ForexData
        fields = '__all__'
class PredictedForexDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictedForexData
        fields = '__all__'