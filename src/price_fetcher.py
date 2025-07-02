import aiohttp
import logging
from datetime import datetime, timedelta

async def fetch_token_price(token_id: str):
    try:
        async with aiohttp.ClientSession() as session:
            current_url = f"https://api.coingecko.com/api/v3/simple/price?ids={token_id}&vs_currencies=usd"
    
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%d-%m-%Y')
            historical_url = f"https://api.coingecko.com/api/v3/coins/{token_id}/history?date={yesterday}"
    
            async with session.get(current_url) as current_resp, \
                session.get(historical_url) as historical_resp:
    
                current_resp.raise_for_status()
                historical_resp.raise_for_status()
    
                current_data = await current_resp.json()
                historical_data = await historical_resp.json()
    
                # Basic validation for expected keys
                if token_id not in current_data or 'usd' not in current_data[token_id] or 'market_data' not in historical_data:
                    logging.error(f"Unexpected API response for {token_id}")
                    return None
    
                return {
                    'current': current_data[token_id]['usd'],
                    'yesterday': historical_data['market_data']['current_price']['usd']
                }
    except aiohttp.ClientError as e:
        logging.error(f"Network error fetching price for {token_id}: {e}")
        return None
    except Exception as e:
        logging.exception(f"An unexpected error occurred in fetch_token_price for {token_id}")
        return None