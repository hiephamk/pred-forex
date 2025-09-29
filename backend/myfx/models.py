from django.db import models

class ForexData(models.Model):
    date = models.DateTimeField(unique=True)
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    class Meta:
        indexes = [
            models.Index(fields=['date']),
        ]
    def __str__(self):
        return f"{self.date} - {self.close}"

class PredictedForexData(models.Model):
    date = models.DateTimeField()
    predicted_close = models.FloatField()

    class Meta:
        db_table = 'predicted_forex_data'
    def __str__(self):
        return f"{self.date} - {self.predicted_close}"