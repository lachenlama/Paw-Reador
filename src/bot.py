import os
import logging
import discord
from dotenv import load_dotenv
from discord.ext import commands
from utils import extract_chunks
from price_fetcher import fetch_token_price

TOKEN_MAP = {
    "bera":"berachain",
    "bgt":"berachain-governance-token",
    "ibgt":"infrared-bgt",
    "lbgt":"liquid-bgt"
}

from rag_model import RAGModel

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DOCS_PATH = os.getenv("DOCS_PATH", "/home/pawreador/Paw-Reador/docs") #in the hosted vps

# Initialize RAG model (indexing at startup)
rag = RAGModel(docs_path=DOCS_PATH)

# Set up Discord bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    logging.info(f"Bot connected as {bot.user}")

@bot.command(name="ask")
async def ask(ctx, *, question: str):
    """Handle !ask command: use RAG to answer user questions."""
    try:
        async with ctx.typing():
            response = rag.query(question, k=5)
            # Extract text from AIMessage or string
            if hasattr(response, 'content'):
                text = response.content
            else:
                text = str(response)
            if '<think>' in text and '</think>' in text:
                text = text.split('</think>')[-1].strip()
            # Discord limits message length; split into chunks if needed
            max_len = 2000
            if len(text) <= max_len:
                await ctx.send(text)
            else:
                # Send in multiple messages
                for i in range(0, len(text), max_len):
                    await ctx.send(text[i:i+max_len])
    except Exception as e:
        logging.exception("Error in ask command")
        await ctx.send("Sorry, I couldn't process your question.")

@bot.command(name="ping")
async def ping(ctx):
    await ctx.send(f"Pong! Latency: {round(bot.latency*1000)}ms")
    
@bot.command(name="price")
async def price(ctx, *, token_name: str):
    """Fetches the price for a given token."""
    token_id = TOKEN_MAP.get(token_name.lower())
    if not token_id:
        await ctx.send(f"Sorry, I don't know the token '{token_name}'. Try one of: {', '.join(TOKEN_MAP.keys())}")
        return
    
    try:
        async with ctx.typing():
            prices = await fetch_token_price(token_id)
            if not prices:
                await ctx.send("Sorry, I couldn't fetch the price data right now.")
                return
            
            change = calculate_change(prices)
            response = (f"**{token_name.upper()} Price**\n"
                        f"Current: `${prices['current']:.4f}`\n"
                        f"Yesterday: `${prices['yesterday']:.4f}`\n"
                        f"24h Change: `{change:.2f}%`")
            await ctx.send(response)
    except Exception as e:
        logging.exception(f"Error fetching price for {token_name}")
        await ctx.send("Sorry, an error occurred while fetching the price.")

#Helper functions
def calculate_change(prices: dict) -> float:
    if prices['yesterday'] == 0:
        return float('inf') if prices['current'] > 0 else 0.0
    return ((prices['current'] - prices['yesterday']) / prices['yesterday']) * 100

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bot.run(TOKEN)