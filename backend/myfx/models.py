from django.db import models

class ForexData(models.Model):
    date = models.DateTimeField(unique=True)
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    
    def __str__(self):
        return f"{self.date} - {self.close}"
