import pandas as pd
from django.core.management.base import BaseCommand
from django.conf import settings
from app.models import Transaction
import os

class Command(BaseCommand):
    help = "Import transactions.csv gerado pelo PaySim"

    def add_arguments(self, parser):
        parser.add_argument('--file', type=str, default=os.getenv("PAYSIM_OUTPUT", settings.PAYSIM_OUTPUT))

    def handle(self, *args, **options):
        path = options['file']
        if not path or not os.path.exists(path):
            self.stderr.write(self.style.ERROR(f"Arquivo não encontrado: {path}"))
            return

        df = pd.read_csv(path)
        self.stdout.write(f"Found {len(df)} rows in {path}")
        objs = []
        batch = 1000
        for _, r in df.iterrows():
            try:
                objs.append(Transaction(
                    step=int(r.get('step', 0)),
                    tx_type=str(r.get('type', '')),
                    amount=float(r.get('amount', 0.0)),
                    nameOrig=str(r.get('nameOrig', '')),
                    nameDest=str(r.get('nameDest', '')),
                    oldbalanceOrg=float(r.get('oldbalanceOrg', 0.0) or 0.0),
                    newbalanceOrig=float(r.get('newbalanceOrig', 0.0) or 0.0),
                    oldbalanceDest=float(r.get('oldbalanceDest', 0.0) or 0.0),
                    newbalanceDest=float(r.get('newbalanceDest', 0.0) or 0.0),
                    isFraud=bool(int(r.get('isFraud', 0) or 0)),
                    isFlaggedFraud=bool(int(r.get('isFlaggedFraud', 0) or 0))
                ))
            except Exception as e:
                self.stderr.write(f"Erro parse linha: {e}")
            if len(objs) >= batch:
                Transaction.objects.bulk_create(objs)
                objs = []
        if objs:
            Transaction.objects.bulk_create(objs)
        self.stdout.write(self.style.SUCCESS("Import completo."))
