import aiohttp
from datetime import datetime, timedelta

async def fetch_token_price(token_id: str):
    async with aiohttp.ClientSession() as session:
        current_url = f"https://api.coingecko.com/api/v3/simple/price?ids={token_id}&vs_currencies=usd"

        yesterday = (datetime.now() - timedelta(days=1)).strftime('%d-%m-%Y')
        historical_url = f"https://api.coingecko.com/api/v3/coins/{token_id}/history?date={yesterday}"

        async with session.get(current_url) as current_resp, \
            session.get(historical_url) as historical_resp:

            current_data = await current_resp.json()
            historical_data = await historical_resp.json()

            return {
                'current': current_data[token_id]['usd'],
                'yesterday': historical_data['market_data']['current_price']['usd']
            }