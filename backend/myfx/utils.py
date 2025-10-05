import os
import pytz
from django.utils.dateparse import parse_datetime
from twelvedata import TDClient
from .models import ForexData
import pandas as pd

def fetch_xauusd(latest_only=True):
    """
    Fetch XAU/USD hourly data from Twelve Data.
    If latest_only=True, fetch only the most recent hour(s) not already in DB.
    Discards incomplete "future" candles.
    """
    api_key = os.environ.get("apiKey_twelvedata")
    if not api_key:
        print("Error: Twelve Data API key not found in environment variables")
        return False

    td = TDClient(apikey=api_key)
    try:
        ts = td.time_series(
            symbol="XAU/USD",
            interval="1h",
            outputsize=1000,
        ).as_json()

        # Normalize response
        if isinstance(ts, tuple):
            values = list(ts)
        elif isinstance(ts, dict) and "values" in ts:
            values = ts["values"]
        else:
            print(f"API error or empty response: {ts}")
            return False

        if not values:
            print("No data available after fetching")
            return False

        # ✅ Drop incomplete (future) candle
        now_utc = pd.Timestamp.now(tz="UTC")
        cleaned_values = []
        for v in values:
            dt_utc = parse_datetime(v["datetime"]).replace(tzinfo=pytz.UTC)
            if dt_utc <= now_utc:
                cleaned_values.append(v)
            else:
                print(f"Dropped incomplete candle at {dt_utc}")

        if not cleaned_values:
            print("All candles were incomplete, nothing to save")
            return False

        # If latest_only, skip if already in DB
        if latest_only:
            latest_entry = ForexData.objects.order_by('-date').first()
            latest_api_time = parse_datetime(cleaned_values[0]['datetime'])
            if latest_api_time and latest_entry and latest_entry.date >= latest_api_time.replace(tzinfo=pytz.UTC):
                print("Latest data already in database")
                return True

        # Save to DB
        for entry in cleaned_values:
            dt_utc = parse_datetime(entry['datetime'])
            if not dt_utc:
                print(f"Invalid datetime format: {entry['datetime']}")
                continue
            dt_utc = dt_utc.replace(tzinfo=pytz.UTC)

            ForexData.objects.update_or_create(
                date=dt_utc,
                defaults={
                    "open": float(entry["open"]),
                    "high": float(entry["high"]),
                    "low": float(entry["low"]),
                    "close": float(entry["close"]),
                }
            )

        print(f"Successfully processed {len(cleaned_values)} records")
        return True

    except Exception as e:
        print(f"Unexpected error processing XAU/USD data: {e}")
        return False