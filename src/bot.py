import os
import logging
import discord
from dotenv import load_dotenv
from discord.ext import commands
from rag_model import RAGModel
from utils import extract_chunks
from price_fetcher import fetch_token_price

TOKEN_MAP = {
    "BERA":"berachain",
    "BGT":"berachain-governance-token",
    "iBGT":"infrared-bgt",
    "lBGT":"liquid-bgt"
}

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
    if "price of" in question.lower():
        token_name = extract_token_name(question)
        token_id = TOKEN_MAP.get(token_name.upper())

        if token_id:
            prices = await fetch_token_price(token_id)
            response = (f"**{token_name} Price**\n"
                        f"Current: ${prices['current']:.2f}\n"
                        f"Yesterday: ${prices['yesterday']:.2f}\n"
                        f"Change: {calculate_change(prices):.2f}%")
            await ctx.send(response)
            return

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

    def on_created(self, event):
        if event.is_directory: return
        chunks = extract_chunks(event.src_path)
        embeds = rag.embeddings.embed_documents([c.page_content for c in chunks])
        rag.db.add_documents(chunks)
        rag.db.persist()
        logging.info(f"Indexed new document: {event.src_path}")

#Helper functions
def extract_token_name(question:str) -> str:
    tokens = question.lower().split()
    if "price" in tokens:
        idx = tokens.index("price")
        return tokens[idx-1] if idx > 0 else "BGT"
    return "BGT"

def calculate_change(prices: dict) -> float:
    return ((prices['current'] - prices['yesterday']) / prices['yesterday']) * 100

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bot.run(TOKEN)