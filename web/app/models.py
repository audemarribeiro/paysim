from django.db import models

class Transaction(models.Model):
    step = models.IntegerField()
    tx_type = models.CharField(max_length=20)
    amount = models.FloatField()
    nameOrig = models.CharField(max_length=64, db_index=True)
    nameDest = models.CharField(max_length=64, db_index=True)
    oldbalanceOrg = models.FloatField(default=0.0)
    newbalanceOrig = models.FloatField(default=0.0)
    oldbalanceDest = models.FloatField(default=0.0)
    newbalanceDest = models.FloatField(default=0.0)
    isFraud = models.BooleanField(default=False)
    isFlaggedFraud = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.step} {self.tx_type} {self.amount}"
