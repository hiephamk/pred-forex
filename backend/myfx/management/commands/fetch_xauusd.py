from django.core.management.base import BaseCommand
from myfx.utils import fetch_xauusd

class Command(BaseCommand):
    help = "Fetch XAU/USD data from Twelve Data and save to database"

    def handle(self, *args, **options):
        fetch_xauusd()
        self.stdout.write(self.style.SUCCESS("Finished fetching XAU/USD data!"))
