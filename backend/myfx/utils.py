import requests
from django.utils.dateparse import parse_datetime
from .models import ForexData

def fetch_xauusd(latest_only=True):
    """
    Fetch XAU/USD daily data from Twelve Data.
    If latest_only=True, fetch only the most recent day(s) not already in DB.
    """
    url = 'https://api.twelvedata.com/time_series'
    params = {
        "symbol": "XAU/USD",
        "interval": "1day",
        "outputsize": 365,  # just fetch last 10 days
        "apikey": "92316069f35c407aa1ebdbd7279fe144",
    }
    response = requests.get(url, params=params)
    data = response.json()

    if 'values' not in data:
        print("Error fetching XAU/USD:", data)
        return

    # Get the latest date in your DB
    latest_in_db = ForexData.objects.order_by('-date').first()
    latest_date = latest_in_db.date if latest_in_db else None

    for entry in data['values']:
        dt = parse_datetime(entry["datetime"])
        if dt is None:
            continue

        # Skip if already in DB
        if latest_date and dt <= latest_date:
            continue

        ForexData.objects.update_or_create(
            date=dt,
            defaults={
                "open": float(entry["open"]),
                "high": float(entry["high"]),
                "low": float(entry["low"]),
                "close": float(entry["close"]),
            }
        )

    print("XAU/USD latest data fetched and saved!")
