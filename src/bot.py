import os
import logging
import discord
from dotenv import load_dotenv
from discord.ext import commands
from rag_model import RAGModel
# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler
from utils import extract_chunks

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DOCS_PATH = os.getenv("DOCS_PATH", "./docs")

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

# File watcher for dynamic document updates\ nclass DocHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory: return
        chunks = extract_chunks(event.src_path)
        embeds = rag.embeddings.embed_documents([c.page_content for c in chunks])
        rag.db.add_documents(chunks)
        rag.db.persist()
        logging.info(f"Indexed new document: {event.src_path}")

# observer = Observer()
# observer.schedule(DocHandler(), path=DOCS_PATH, recursive=False)
# observer.start()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bot.run(TOKEN)