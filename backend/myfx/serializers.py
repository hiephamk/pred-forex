from rest_framework import serializers
from .models import ForexData


class forexDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = ForexData
        fields = '__all__'