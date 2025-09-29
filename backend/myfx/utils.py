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
            outputsize=1 if latest_only else 48
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









# from twelvedata import TDClient
# from django.utils.dateparse import parse_datetime
# from django.utils import timezone
# from .models import ForexData
# import os
# import pytz
# from datetime import datetime
#
# def fetch_xauusd(latest_only=True):
#     """
#     Fetch XAU/USD hourly data from Twelve Data using the official library.
#     If latest_only=True, fetch only the most recent hour(s) not already in DB.
#     Skips future timestamps.
#     Returns True on success, False on failure.
#     """
#     api_key = os.environ.get("apiKey_twelvedata")
#     if not api_key:
#         print("Error: Twelve Data API key not found")
#         return False
#
#     # Initialize the client
#     td = TDClient(apikey=api_key)
#
#     # Calculate current UTC time for end_date
#     now_utc = datetime.now(pytz.UTC)
#     end_date_str = now_utc.strftime("%Y-%m-%d %H:%M:%S")
#
#     # Parameters for time_series
#     params = {
#         "symbol": "XAU/USD",
#         "interval": "1h",
#         "outputsize": 2 if latest_only else 48,  # Fetch 2 for latest to ensure a completed bar
#         "end_date": end_date_str,  # Limit to current time to avoid future bars
#         "timezone": "UTC",  # Ensure UTC timestamps
#         "order": "desc",  # Newest first
#     }
#
#     try:
#         # Fetch time series data
#         ts = td.time_series(**params)
#         data = ts.as_json()  # Get as JSON for easy processing
#
#         # Log raw response for debugging
#         print("Raw API response:", data)
#
#         # Check if API call was successful
#         if data.get('status') != 'ok':
#             print(f"API error: {data.get('message', 'Unknown error')}")
#             return False
#
#         values = data.get('values', [])
#         if not values:
#             print("No data returned in API response")
#             return False
#
#         # Get current UTC time for filtering
#         now_utc = timezone.now().astimezone(pytz.UTC)
#
#         # If latest_only, check if the latest data is already in the DB
#         if latest_only:
#             latest_entry = ForexData.objects.order_by('-date').first()
#             latest_api_time = parse_datetime(values[0]['datetime'])
#             if latest_api_time:
#                 latest_api_time = latest_api_time.replace(tzinfo=pytz.UTC)
#                 if latest_entry and latest_entry.date >= latest_api_time:
#                     print("Latest data already in database")
#                     return True
#
#         # Process time series data
#         processed_count = 0
#         for entry in values:
#             dt_utc = parse_datetime(entry['datetime'])
#             if not dt_utc:
#                 print(f"Invalid datetime format: {entry['datetime']}")
#                 continue
#
#             dt_utc = dt_utc.replace(tzinfo=pytz.UTC)
#
#             # Skip future timestamps
#             if dt_utc > now_utc:
#                 print(f"Skipping future timestamp: {dt_utc}")
#                 continue
#
#             # Update or create ForexData entry
#             ForexData.objects.update_or_create(
#                 date=dt_utc,
#                 defaults={
#                     "open": float(entry["open"]),
#                     "high": float(entry["high"]),
#                     "low": float(entry["low"]),
#                     "close": float(entry["close"]),
#                 }
#             )
#             processed_count += 1
#
#         print(f"Successfully processed {processed_count} records")
#         return True
#
#     except Exception as e:
#         print(f"Error fetching or processing XAU/USD data: {e}")
#         return False


# import requests
# from django.utils.dateparse import parse_datetime
# from django.utils import timezone
# from twelvedata import TDClient
# from .models import ForexData
# import os
# import pytz
#
# def fetch_xauusd(latest_only=True):
#     """
#     Fetch XAU/USD hourly data from Twelve Data.
#     If latest_only=True, fetch only the most recent hour(s) not already in DB.
#     Returns True on success, False on failure.
#     """
#     # Validate API key
#     api_key = os.environ.get("apiKey_twelvedata")
#     if not api_key:
#         print("Error: Twelve Data API key not found in environment variables")
#         return False
#     td = TDClient(apikey=api_key)
#     # url = 'https://api.twelvedata.com/time_series'
#     params = {
#         "symbol": "XAU/USD",
#         "interval": "1h",
#         "outputsize": 1 if latest_only else 48,
#         "apikey": api_key,
#     }
#
#     try:
#         # response = requests.get(url, params=params)
#         # response.raise_for_status()
#         # data = response.json()
#         ts = td.time_series(symbol=symbol, interval=interval, outputsize=outputsize).as_pandas()
#
#         # Check if API call was successful
#         if data.get('status') != 'ok':
#             print(f"API error: {data.get('message', 'Unknown error')}")
#             return False
#
#         # Check if values exist in response
#         values = data.get('values', [])
#         if not values:
#             print("No data returned in API response")
#             return False
#
#         # If latest_only, check if the latest data is already in the DB
#         if latest_only:
#             latest_entry = ForexData.objects.order_by('-date').first()
#             latest_api_time = parse_datetime(values[0]['datetime'])
#             if latest_api_time and latest_entry and latest_entry.date >= latest_api_time.replace(tzinfo=pytz.UTC):
#                 print("Latest data already in database")
#                 return True
#
#         # Process time series data
#         for entry in values:
#             dt_utc = parse_datetime(entry['datetime'])
#             if not dt_utc:
#                 print(f"Invalid datetime format: {entry['datetime']}")
#                 continue
#
#             dt_utc = dt_utc.replace(tzinfo=pytz.UTC)
#
#             # Update or create ForexData entry
#             ForexData.objects.update_or_create(
#                 date=dt_utc,
#                 defaults={
#                     "open": float(entry["open"]),
#                     "high": float(entry["high"]),
#                     "low": float(entry["low"]),
#                     "close": float(entry["close"]),
#                 }
#             )
#
#         print(f"Successfully processed {len(values)} records")
#         return True
#
#     except requests.HTTPError as e:
#         print(f"HTTP error fetching XAU/USD data: {e} (Status code: {e.response.status_code})")
#         return False
#     except requests.RequestException as e:
#         print(f"Error fetching XAU/USD data: {e}")
#         return False
#     except Exception as e:
#         print(f"Unexpected error processing XAU/USD data: {e}")
#         return False
# #
#
#
#
# # import requests
# # from django.utils.dateparse import parse_datetime
# # from django.utils import timezone
# # from .models import ForexData
# # import os
# # from datetime import datetime as dt, timezone as dt_timezone
# # import pytz
# #
# # def fetch_xauusd(latest_only=True):
# #     """
# #     Fetch XAU/USD hourly data from Twelve Data.
# #     If latest_only=True, fetch only the most recent hour(s) not already in DB.
# #     """
# #     url = 'https://api.twelvedata.com/time_series'
# #     params = {
# #         "symbol": "XAU/USD",
# #         "interval": "1h",
# #         "outputsize": 48 if not latest_only else 1,
# #         "apikey": os.environ.get("apiKey_twelvedata"),
# #     }
# #
# #     try:
# #         response = requests.get(url, params=params)
# #         response.raise_for_status()
# #         data = response.json()
# #
# #         # Check if API call was successful
# #         if data.get('status') != 'ok':
# #             print(f"API error: {data.get('message', 'Unknown error')}")
# #             return
# #
# #         # Process time series data
# #         for entry in data.get('values', []):
# #             ForexData.objects.update_or_create(
# #                 # date=dt_utc,
# #                 defaults={
# #                     "open": float(entry["open"]),
# #                     "high": float(entry["high"]),
# #                     "low": float(entry["low"]),
# #                     "close": float(entry["close"]),
# #                 }
# #             )
# #             # Convert timestamp to UTC datetime
# #             # dt_utc = parse_datetime(entry['datetime'])
# #             # if dt_utc:
# #             #     dt_utc = dt_utc.replace(tzinfo=pytz.UTC)
# #             #
# #             #     # Update or create ForexData entry
# #             #     ForexData.objects.update_or_create(
# #             #         date=dt_utc,
# #             #         defaults={
# #             #             "open": float(entry["open"]),
# #             #             "high": float(entry["high"]),
# #             #             "low": float(entry["low"]),
# #             #             "close": float(entry["close"]),
# #             #         }
# #             #     )
# #
# #     except requests.RequestException as e:
# #         print(f"Error fetching XAU/USD data: {e}")
# #     except Exception as e:
# #         print(f"Unexpected error processing XAU/USD data: {e}")