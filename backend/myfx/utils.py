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
        "outputsize": 30,  # just fetch last 10 days
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

"""
version2 - get current data
"""
# import requests
# from django.utils.dateparse import parse_datetime
# from .models import ForexData
# from datetime import datetime
# import logging

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def fetch_xauusd(latest_only=True, min_data_points=SEQUENCE_LENGTH + 1):
#     """
#     Fetch XAU/USD daily data from Twelve Data.
#     Args:
#         latest_only (bool): If True, fetch only data newer than the latest in DB.
#         min_data_points (int): Minimum number of data points required (default: SEQUENCE_LENGTH + 1).
#     Returns:
#         bool: True if data was successfully fetched and saved, False otherwise.
#     """
#     try:
#         url = 'https://api.twelvedata.com/time_series'
#         params = {
#             "symbol": "XAU/USD",
#             "interval": "1day",
#             "outputsize": 30 if latest_only else max(30, min_data_points * 2),  # Ensure enough data
#             "apikey": "92316069f35c407aa1ebdbd7279fe144",
#         }

#         response = requests.get(url, params=params, timeout=10)
#         response.raise_for_status()  # Raise exception for bad status codes
#         data = response.json()

#         if 'values' not in data or not data['values']:
#             logger.error(f"Error fetching XAU/USD: {data.get('message', 'No values returned')}")
#             return False

#         # Get the latest date in the database
#         latest_in_db = ForexData.objects.order_by('-date').first()
#         latest_date = latest_in_db.date if latest_in_db else None

#         saved_count = 0
#         for entry in data['values']:
#             dt = parse_datetime(entry["datetime"])
#             if dt is None:
#                 logger.warning(f"Invalid datetime format: {entry['datetime']}")
#                 continue

#             # Ensure date is timezone-naive for consistency with Django
#             dt = dt.replace(tzinfo=None) if dt.tzinfo else dt

#             # Skip if already in DB and fetching only latest
#             if latest_only and latest_date and dt.date() <= latest_date:
#                 continue

#             # Save or update record
#             ForexData.objects.update_or_create(
#                 date=dt.date(),
#                 defaults={
#                     "open": float(entry["open"]),
#                     "high": float(entry["high"]),
#                     "low": float(entry["low"]),
#                     "close": float(entry["close"]),
#                     "volume": float(entry.get("volume", 0))  # Volume may not be available
#                 }
#             )
#             saved_count += 1

#         logger.info(f"XAU/USD: Fetched and saved {saved_count} new records.")
#         return saved_count > 0

#     except requests.exceptions.RequestException as e:
#         logger.error(f"Network error fetching XAU/USD: {e}")
#         return False
#     except Exception as e:
#         logger.error(f"Unexpected error fetching XAU/USD: {e}")
#         return False
