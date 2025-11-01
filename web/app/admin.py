from django.contrib import admin
from .models import Transaction

@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    list_display = ("id", "step", "tx_type", "amount", "nameOrig", "nameDest", "isFraud")
    list_filter = ("tx_type", "isFraud", "isFlaggedFraud")
    search_fields = ("nameOrig", "nameDest")
