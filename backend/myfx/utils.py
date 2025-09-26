import requests
from django.utils.dateparse import parse_datetime
from django.utils import timezone
from .models import ForexData
import os
from datetime import datetime as dt, timezone as dt_timezone
import pytz

def fetch_xauusd(latest_only=True):
    """
    Fetch XAU/USD hourly data from Twelve Data.
    If latest_only=True, fetch only the most recent hour(s) not already in DB.
    """
    url = 'https://api.twelvedata.com/time_series'
    params = {
        "symbol": "XAU/USD",
        "interval": "1h",
        "outputsize": 960 if not latest_only else 200,
        "apikey": os.environ.get("apiKey_twelvedata"),
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Check if API call was successful
        if data.get('status') != 'ok':
            print(f"API error: {data.get('message', 'Unknown error')}")
            return

        # Process time series data
        for entry in data.get('values', []):
            # Convert timestamp to UTC datetime
            dt_utc = parse_datetime(entry['datetime'])
            if dt_utc:
                dt_utc = dt_utc.replace(tzinfo=pytz.UTC)
                
                # Update or create ForexData entry
                ForexData.objects.update_or_create(
                    date=dt_utc,
                    defaults={
                        "open": float(entry["open"]),
                        "high": float(entry["high"]),
                        "low": float(entry["low"]),
                        "close": float(entry["close"]),
                    }
                )

    except requests.RequestException as e:
        print(f"Error fetching XAU/USD data: {e}")
    except Exception as e:
        print(f"Unexpected error processing XAU/USD data: {e}")